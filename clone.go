package main

import (
	"compress/flate"
	"errors"
	"flag"
	"io/ioutil"
	"log"
	"math/rand"
	"path/filepath"
	"strings"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

// CloneFlags are flags for behavior cloning.
type CloneFlags struct {
	GeneralFlags

	Dir        string
	Batch      int
	Validation string
	L2Reg      float64
}

// Add adds the flags to a flag set.
func (c *CloneFlags) Add(fs *flag.FlagSet) {
	c.GeneralFlags.Add(fs)
	fs.StringVar(&c.Dir, "dir", "", "training sample directory")
	fs.StringVar(&c.Validation, "validation", "", "validation sample directory")
	fs.IntVar(&c.Batch, "batch", 16, "batch size")
	fs.Float64Var(&c.L2Reg, "l2reg", 0, "L2 regularization")
}

// Clone performs behavior cloning.
func Clone(c anyvec.Creator, args []string) {
	rand.Seed(time.Now().UnixNano())

	fs := flag.NewFlagSet("clone", flag.ExitOnError)
	flags := &CloneFlags{}
	flags.Add(fs)
	fs.Parse(args)

	spec := MustSpecForName(flags.EnvName)
	policy, _ := LoadOrMakeAgent(c, spec, flags.PolicyFile, "", false)

	samples, err := ReadSampleList(flags.Dir)
	if err != nil {
		essentials.Die(err)
	}
	var validation SampleList
	if flags.Validation != "" {
		validation, err = ReadSampleList(flags.Validation)
		if err != nil {
			essentials.Die(err)
		}
	}
	trainer := &Trainer{
		Policy: func(seq lazyseq.Rereader) lazyseq.Rereader {
			return ApplyBlock(seq, policy)
		},
		Spec:   spec,
		Params: anynet.AllParameters(policy),
		L2Reg:  flags.L2Reg,
	}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     trainer,
		Gradienter:  trainer,
		Transformer: &anysgd.Adam{},
		Samples:     samples,
		Rater:       anysgd.ConstRater(0.001),
		BatchSize:   flags.Batch,
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iteration %d: cost=%v", iter, trainer.LastCost)
			if iter%4 == 0 && len(validation) > 0 {
				anysgd.Shuffle(validation)
				batchSize := essentials.MinInt(validation.Len(), flags.Batch)
				vbatch, err := trainer.Fetch(validation.Slice(0, batchSize))
				if err != nil {
					essentials.Die(err)
				}
				cost := trainer.TotalCost(vbatch)
				log.Printf("iteration %d: val_cost=%f", iter,
					anyvec.Sum(cost.Output()))
			}
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())
	serializer.SaveAny(flags.PolicyFile, policy)
}

// A Batch stores a batch of demonstrations in a format
// suitable for supervised learning.
type Batch struct {
	Observations lazyseq.Tape
	Actions      lazyseq.Tape
}

// A SampleList is a list of recording directories for
// supervised training.
type SampleList []string

// ReadSampleList reads a directory full of recordings.
func ReadSampleList(dir string) (list SampleList, err error) {
	defer essentials.AddCtxTo("read sample list", &err)
	listing, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, item := range listing {
		if strings.HasPrefix(item.Name(), "recording_") && item.IsDir() {
			list = append(list, filepath.Join(dir, item.Name()))
		}
	}
	return
}

// Len returns the number of samples.
func (s SampleList) Len() int {
	return len(s)
}

// Swap swaps two samples.
func (s SampleList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Slice generates a copy of a subslice.
func (s SampleList) Slice(i, j int) anysgd.SampleList {
	return append(SampleList{}, s[i:j]...)
}

// A Trainer is used to train a policy to immitate
// demonstration data.
type Trainer struct {
	Policy func(lazyseq.Rereader) lazyseq.Rereader
	Spec   *EnvSpec
	Params []*anydiff.Var
	L2Reg  float64

	// LastCost is set to the cost after every gradient
	// computation.
	LastCost anyvec.Numeric
}

// Fetch produces a *Batch for a subset of recordings.
// The s argument must be a SampleList.
// The batch may not be empty.
func (t *Trainer) Fetch(s anysgd.SampleList) (batch anysgd.Batch, err error) {
	defer essentials.AddCtxTo("fetch batch", &err)
	if s.Len() == 0 {
		return nil, errors.New("empty batch")
	}
	recordingDirs := s.(SampleList)
	actors := make([]Actor, s.Len())
	recordings := make([]*muniverse.Recording, s.Len())
	for i, recordingDir := range recordingDirs {
		recording, err := muniverse.OpenRecording(recordingDir)
		if err != nil {
			return nil, err
		}
		recordings[i] = recording
		actors[i] = t.Spec.MakeActor()
		actors[i].Reset()
	}
	inTape, inWriter := lazyseq.CompressedUint8Tape(flate.DefaultCompression)
	outTape, outWriter := lazyseq.ReferenceTape()
	defer close(inWriter)
	defer close(outWriter)
	obsJoiners := make([]*ObsJoiner, s.Len())
	for i := range obsJoiners {
		obsJoiners[i] = &ObsJoiner{HistorySize: t.Spec.HistorySize}
	}
	for i := 0; true; i++ {
		var inVecs []anyvec.Vector
		var outVecs []anyvec.Vector
		var present []bool
		for j, recording := range recordings {
			pres := i < recording.NumSteps()
			present = append(present, pres)
			if !pres {
				continue
			}
			obs, err := recording.ReadObs(i)
			if err != nil {
				return nil, err
			}
			step, err := recording.ReadStep(i)
			if err != nil {
				return nil, err
			}
			vec, err := t.Spec.Observer.ObsVec(t.creator(), obs)
			if err != nil {
				return nil, err
			}
			if i == 0 {
				obsJoiners[j].Reset(vec)
			}
			inVecs = append(inVecs, obsJoiners[j].Step(vec))
			vec = actors[j].Vectorize(t.creator(), step.Events)
			outVecs = append(outVecs, vec)
		}
		if len(inVecs) == 0 {
			break
		}
		inWriter <- &anyseq.Batch{
			Packed:  t.creator().Concat(inVecs...),
			Present: present,
		}
		outWriter <- &anyseq.Batch{
			Packed:  t.creator().Concat(outVecs...),
			Present: present,
		}
	}
	return &Batch{
		Observations: inTape,
		Actions:      outTape,
	}, nil
}

// TotalCost computes the average negative log-likelihood
// for actions in the *Batch.
func (t *Trainer) TotalCost(batch anysgd.Batch) anydiff.Res {
	b := batch.(*Batch)
	inSeq := lazyseq.TapeRereader(t.creator(), b.Observations)
	desired := lazyseq.TapeRereader(t.creator(), b.Actions)
	actual := t.Policy(inSeq)
	space := t.Spec.MakeActor().ActionSpace()
	logLikelihood := lazyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		return space.LogProb(v[0], v[1].Output(), n)
	}, actual, desired)
	return anydiff.Scale(lazyseq.Mean(logLikelihood), t.creator().MakeNumeric(-1))
}

// Gradient computes the gradient for the *Batch.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	grad, lc := anysgd.CosterGrad(t, b, t.Params)
	t.LastCost = lc
	if t.L2Reg != 0 {
		for _, param := range t.Params {
			regTerm := param.Output().Copy()
			regTerm.Scale(regTerm.Creator().MakeNumeric(t.L2Reg))
			grad[param].Add(regTerm)
		}
	}
	return grad
}

func (t *Trainer) creator() anyvec.Creator {
	return t.Params[0].Vector.Creator()
}
