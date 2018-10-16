package main

import (
	"fmt"
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/serializer"
)

// LoadOrMakeAgent creates or loads a policy and a critic
// from the specified files.
// It logs what it is doing so the user knows whether an
// RNN was created or not.
//
// If needsCritic is false, then no critic is loaded.
func LoadOrMakeAgent(creator anyvec.Creator, spec *EnvSpec, policyPath,
	criticPath string, needsCritic bool) (policy, critic anyrnn.Block) {
	if err := serializer.LoadAny(policyPath, &policy); err != nil {
		log.Println("Creating new policy...")
		policy = MakePolicy(creator, spec)
	} else {
		log.Println("Loaded policy.")
	}
	if needsCritic {
		if err := serializer.LoadAny(criticPath, &critic); err != nil {
			log.Println("Creating new critic...")
			critic = MakeCritic(creator)
		} else {
			log.Println("Loaded critic.")
		}
	}
	return
}

// MakePolicy creates a new policy RNN which is compatible
// with the environment specification.
func MakePolicy(c anyvec.Creator, e *EnvSpec) anyrnn.Block {
	w, h, d := e.Observer.ObsSize()
	markup := fmt.Sprintf(`
		Input(w=%d, h=%d, d=%d)
		Linear(scale=0.01)
		Conv(w=8, h=8, n=32, sx=4, sy=4)
		ReLU
		Conv(w=4, h=4, n=64, sx=2, sy=2)
		ReLU
		Conv(w=3, h=3, n=64, sx=1, sy=1)
		ReLU
		FC(out=512)
		ReLU
	`, w, h, d*(1+e.HistorySize))
	convNet, err := anyconv.FromMarkup(c, markup)
	if err != nil {
		panic(err)
	}
	return anyrnn.Stack{
		&anyrnn.LayerBlock{
			Layer: append(
				setupVisionLayers(convNet.(anynet.Net)),
				anynet.NewFCZero(c, 256, e.MakeActor().ParamLen()),
			),
		},
	}
}

// MakeCritic creates a critic block for A3C.
func MakeCritic(c anyvec.Creator) anyrnn.Block {
	return &anyrnn.LayerBlock{
		Layer: anynet.NewFC(c, 256, 1),
	}
}

// MakeAgent creates an A3C agent for the RNN blocks.
func MakeAgent(c anyvec.Creator, e *EnvSpec, policy,
	critic anyrnn.Block) *anya3c.Agent {
	policyNet := policy.(anyrnn.Stack)[0].(*anyrnn.LayerBlock).Layer.(anynet.Net)
	baseNet := policyNet[:len(policyNet)-1]
	actorNet := policyNet[len(policyNet)-1:]
	return &anya3c.Agent{
		Base:        &anyrnn.LayerBlock{Layer: baseNet},
		Actor:       &anyrnn.LayerBlock{Layer: actorNet},
		Critic:      critic,
		ActionSpace: e.MakeActor().ActionSpace(),
	}
}

// ApplyBlock applies the block in a memory-efficient
// manner.
func ApplyBlock(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
	switch b := b.(type) {
	case anyrnn.Stack:
		if len(b) != 1 {
			panic("expected one entry")
		}
		return ApplyBlock(seq, b[0])
	case *anyrnn.LayerBlock:
		return lazyseq.Map(seq, b.Layer.Apply)
	default:
		panic(fmt.Sprintf("unexpected block type: %T", b))
	}
}

// DecomposeAgent decomposes an A3C agent into the policy
// and the critic.
func DecomposeAgent(a *anya3c.Agent) (policy, critic anyrnn.Block) {
	critic = a.Critic
	baseNet := a.Base.(*anyrnn.LayerBlock).Layer.(anynet.Net)
	actorNet := a.Actor.(*anyrnn.LayerBlock).Layer.(anynet.Net)
	policyNet := append(append(anynet.Net{}, baseNet...), actorNet...)
	policy = anyrnn.Stack{&anyrnn.LayerBlock{Layer: policyNet}}
	return
}

func setupVisionLayers(net anynet.Net) anynet.Net {
	for _, layer := range net {
		projectOutSolidColors(layer)
	}
	return net
}

func projectOutSolidColors(layer anynet.Layer) {
	switch layer := layer.(type) {
	case *anyconv.Conv:
		filters := layer.Filters.Vector
		inDepth := layer.InputDepth
		numFilters := layer.FilterCount
		filterSize := filters.Len() / numFilters
		for i := 0; i < numFilters; i++ {
			filter := filters.Slice(i*filterSize, (i+1)*filterSize)

			// Compute the mean for each input channel.
			negMean := anyvec.SumRows(filter, inDepth)
			negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(filterSize/inDepth)))
			anyvec.AddRepeated(filter, negMean)
		}
	case *anynet.FC:
		negMean := anyvec.SumCols(layer.Weights.Vector, layer.OutCount)
		negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(layer.InCount)))
		anyvec.AddChunks(layer.Weights.Vector, negMean)
	}
}
