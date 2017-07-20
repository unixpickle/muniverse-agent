package main

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
)

// An Observer converts raw muniverse observations into
// tensor data.
type Observer interface {
	// ObsSize returns the output tensor size.
	ObsSize() (width, height, depth int)

	// ObsVec creates a packed tensor.
	ObsVec(c anyvec.Creator, obs muniverse.Obs) (anyvec.Vector, error)
}

// A DownsampleObserver downsamples an image and converts
// it to a grayscale tensor.
type DownsampleObserver struct {
	StrideX int
	StrideY int

	InWidth  int
	InHeight int

	Color bool
}

// ObsSize returns the output tensor size.
func (d *DownsampleObserver) ObsSize() (width, height, depth int) {
	depth = 1
	if d.Color {
		depth = 3
	}
	width = d.InWidth / d.StrideX
	if d.InWidth%d.StrideX != 0 {
		width += 1
	}
	height = d.InHeight / d.StrideY
	if d.InHeight%d.StrideY != 0 {
		height += 1
	}
	return
}

// ObsVec downsamples the image and converts it to a
// tensor.
func (d *DownsampleObserver) ObsVec(c anyvec.Creator,
	obs muniverse.Obs) (anyvec.Vector, error) {
	buffer, _, _, err := muniverse.RGB(obs)
	if err != nil {
		return nil, err
	}
	w, h, _ := d.ObsSize()
	data := make([]float64, 0, w*h)
	for y := 0; y < d.InHeight; y += d.StrideY {
		for x := 0; x < d.InWidth; x += d.StrideX {
			sourceIdx := (y*d.InWidth + x) * 3
			if d.Color {
				for d := 0; d < 3; d++ {
					data = append(data, float64(buffer[sourceIdx+d]))
				}
			} else {
				var value float64
				for d := 0; d < 3; d++ {
					value += float64(buffer[sourceIdx+d])
				}
				data = append(data, essentials.Round(value/3))
			}
		}
	}
	return c.MakeVectorData(c.MakeNumericList(data)), nil
}

// An ObsJoiner joins together a history of observations.
type ObsJoiner struct {
	HistorySize int

	hist []anyvec.Vector
}

// Reset fills the history with the given frame.
func (o *ObsJoiner) Reset(obs anyvec.Vector) {
	o.hist = make([]anyvec.Vector, o.HistorySize)
	for i := range o.hist {
		o.hist[i] = obs.Copy()
	}
}

// Step updates the history with the new observation and
// returns the latest joined observation.
func (o *ObsJoiner) Step(obs anyvec.Vector) anyvec.Vector {
	joined := joinFrames(o.hist, obs)
	if len(o.hist) > 0 {
		copy(o.hist, o.hist[1:])
		o.hist[len(o.hist)-1] = obs.Copy()
	}
	return joined
}

func joinFrames(hist []anyvec.Vector, current anyvec.Vector) anyvec.Vector {
	allFrames := append(append([]anyvec.Vector{}, hist...), current)
	joined := current.Creator().Concat(allFrames...)
	res := joined.Creator().MakeVector(joined.Len())
	anyvec.Transpose(joined, res, len(allFrames))
	return res
}
