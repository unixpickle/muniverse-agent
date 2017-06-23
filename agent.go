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

// MakeAgent creates a new agent which is compatible with
// the environment specification.
func MakeAgent(c anyvec.Creator, e *EnvSpec) *anya3c.Agent {
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
	`, w, h, d*2)
	convNet, err := anyconv.FromMarkup(c, markup)
	if err != nil {
		panic(err)
	}
	return &anya3c.Agent{
		Base: &anyrnn.LayerBlock{
			Layer: setupVisionLayers(convNet.(anynet.Net)),
		},
		Actor: &anyrnn.LayerBlock{
			Layer: anynet.NewFCZero(c, 256, e.MakeActor().ParamLen()),
		},
		Critic: &anyrnn.LayerBlock{
			Layer: anynet.NewFCZero(c, 256, 1),
		},
		ActionSpace: e.MakeActor().ActionSpace(),
	}
}

// ApplyPolicy applies the policy in a memory-efficient
// manner.
func ApplyPolicy(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
	out := lazyrnn.FixedHSM(30, true, seq, b)
	return lazyseq.Lazify(lazyseq.Unlazify(out))
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