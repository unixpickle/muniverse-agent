package main

import (
	"fmt"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
)

// MakePolicy creates a new policy RNN which is compatible
// with the environment specification.
func MakePolicy(c anyvec.Creator, e *EnvSpec) anyrnn.Block {
	w, h, d := e.Observer.ObsSize()
	markup := fmt.Sprintf(`
		Input(w=%d, h=%d, d=%d)
		Linear(scale=0.01)
		Conv(w=4, h=4, n=16, sx=2, sy=2)
		Tanh
		Conv(w=4, h=4, n=32, sx=2, sy=2)
		Tanh
		FC(out=256)
		Tanh
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

// ApplyPolicy applies the policy in a memory-efficient
// manner.
func ApplyPolicy(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
	out := lazyrnn.FixedHSM(30, true, seq, b)
	return lazyseq.Lazify(lazyseq.Unlazify(out))
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
