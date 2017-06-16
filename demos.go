package main

import (
	"compress/flate"
	"errors"
	"io/ioutil"
	"log"
	"path/filepath"
	"strings"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

// SupervisedTrain performs supervised training on
// demonstration data.
func SupervisedTrain(flags Flags, spec *EnvSpec, policy anyrnn.Block) {
	samples, err := ReadSampleList(flags.DemosDir)
	if err != nil {
		essentials.Die(err)
	}
	trainer := &Trainer{
		Policy: func(seq lazyseq.Rereader) lazyseq.Rereader {
			return ApplyPolicy(seq, policy)
		},
		MakeActor: spec.MakeActor,
		Observer:  spec.Observer,
		Params:    anynet.AllParameters(policy),
	}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     trainer,
		Gradienter:  trainer,
		Transformer: &anysgd.Adam{},
		Samples:     samples,
		Rater:       anysgd.ConstRater(0.001),
		BatchSize:   flags.DemoBatch,
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iteration %d: cost=%v", iter, trainer.LastCost)
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())
	serializer.SaveAny(flags.OutFile, policy)
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
	Policy    func(lazyseq.Rereader) lazyseq.Rereader
	MakeActor func() Actor
	Observer  Observer
	Params    []*anydiff.Var

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
		actors[i] = t.MakeActor()
		actors[i].Reset()
	}
	inTape, inWriter := lazyseq.CompressedUint8Tape(flate.DefaultCompression)
	outTape, outWriter := lazyseq.ReferenceTape()
	defer close(inWriter)
	defer close(outWriter)
	lastFrames := make([]anyvec.Vector, s.Len())
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
			vec, err := t.Observer.ObsVec(t.creator(), obs)
			if err != nil {
				return nil, err
			}
			if lastFrames[j] == nil {
				lastFrames[j] = vec.Copy()
			}
			inVecs = append(inVecs, joinFrames(lastFrames[j], vec))
			lastFrames[j] = vec
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
	space := t.MakeActor().ActionSpace()
	logLikelihood := lazyseq.MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		return space.LogProb(v[0], v[1].Output(), n)
	}, actual, desired)
	return anydiff.Scale(lazyseq.Mean(logLikelihood), t.creator().MakeNumeric(-1))
}

// Gradient computes the gradient for the *Batch.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	grad, lc := anysgd.CosterGrad(t, b, t.Params)
	t.LastCost = lc
	return grad
}

func (t *Trainer) creator() anyvec.Creator {
	return t.Params[0].Vector.Creator()
}
