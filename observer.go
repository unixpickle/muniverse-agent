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
}

// ObsSize returns the output tensor size.
func (d *DownsampleObserver) ObsSize() (width, height, depth int) {
	depth = 1
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
			var value float64
			for d := 0; d < 3; d++ {
				value += float64(buffer[sourceIdx+d])
			}
			data = append(data, essentials.Round(value/3))
		}
	}
	return c.MakeVectorData(c.MakeNumericList(data)), nil
}
