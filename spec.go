package main

import (
	"time"

	"github.com/unixpickle/muniverse"
)

// An EnvSpec contains information about an environment
// that makes it possible to train an agent on said
// environment.
type EnvSpec struct {
	*muniverse.EnvSpec

	Observer  Observer
	MakeActor func() Actor

	// Training hyper-parameters.
	DiscountFactor float64
	FrameTime      time.Duration
	BatchSize      int
}

var EnvSpecs = []*EnvSpec{
	StandardKeySpec("Knightower-v0", true, 0.9, time.Second/8, 512),
	StandardKeySpec("KumbaKarate-v0", true, 0.7, time.Second/10, 512),
	StandardKeySpec("PenguinSkip-v0", true, 0.7, time.Second/5, 512),
	StandardKeySpec("CartoonFlight-v0", true, 0.9, time.Second/8, 512),
	StandardKeySpec("CartoonFlight-v1", true, 0.9, time.Second/8, 512),
	StandardTapSpec("DontCrash-v0", true, 0.9, time.Second/10, 512),
	StandardTapSpec("RabbitPunch-v0", true, 0.9, time.Second/8, 512),
	StandardKeySpec("Twins-v0", false, 0.98, time.Second/10, 2048),
	StandardKeySpec("RedHead-v0", false, 0.98, time.Second/10, 2048),
}

// SpecForName finds a specification in EnvSpecs.
func SpecForName(name string) *EnvSpec {
	for _, s := range EnvSpecs {
		if s.Name == name {
			return s
		}
	}
	return nil
}

// StandardKeySpec generates an *EnvSpec with a common
// Observer and Actor.
//
// The environment must rely entirely on keyboard input.
func StandardKeySpec(name string, noHold bool, discount float64,
	frameTime time.Duration, batchSize int) *EnvSpec {
	raw := muniverse.SpecForName(name)
	if raw == nil {
		panic("no environment: " + name)
	}
	return &EnvSpec{
		EnvSpec: raw,
		Observer: &DownsampleObserver{
			StrideX:  4,
			StrideY:  4,
			InWidth:  raw.Width,
			InHeight: raw.Height,
		},
		MakeActor: func() Actor {
			return &KeyActor{
				Keys:   raw.KeyWhitelist,
				NoHold: noHold,
			}
		},

		DiscountFactor: discount,
		FrameTime:      frameTime,
		BatchSize:      batchSize,
	}
}

// StandardTapSpec is like StandardKeySpec, except with
// TapActor instead of KeyActor.
func StandardTapSpec(name string, noHold bool, discount float64,
	frameTime time.Duration, batchSize int) *EnvSpec {
	res := StandardKeySpec(name, noHold, discount, frameTime, batchSize)
	res.MakeActor = func() Actor {
		return &TapActor{
			Width:  res.Width,
			Height: res.Height,
			NoHold: noHold,
		}
	}
	return res
}
