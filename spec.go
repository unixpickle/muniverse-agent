package main

import (
	"time"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
)

// An EnvSpec contains information about an environment
// that makes it possible to train an agent on said
// environment.
type EnvSpec struct {
	*muniverse.EnvSpec

	// Wrap, if non-nil, wraps every environment when
	// it is created.
	Wrap func(e muniverse.Env) muniverse.Env

	Observer  Observer
	MakeActor func() Actor

	// Training hyper-parameters.
	DiscountFactor float64
	FrameTime      time.Duration
	BatchSize      int

	// If non-zero, used to scale down the rewards to a
	// reasonable regime (e.g. to be close to 0-1).
	RewardScale float64

	// HistorySize is the number of previous observations
	// to feed into the network in addition to the current
	// observation.
	HistorySize int
}

var EnvSpecs = []*EnvSpec{
	StandardKeySpec("Knightower-v0", true, 0.9, time.Second/8, 512),
	StandardKeySpec("KumbaKarate-v0", true, 0.7, time.Second/10, 512),
	StandardKeySpec("PenguinSkip-v0", true, 0.7, time.Second/5, 512),
	StandardKeySpec("TRexRunner-v0", true, 0.98, time.Second/10, 512),
	StandardTapSpec("DontCrash-v0", true, 0.9, time.Second/10, 512),
	StandardTapSpec("RabbitPunch-v0", true, 0.9, time.Second/8, 512),
	StandardTapSpec("Babel-v0", true, 0.98, time.Second/10, 1024),
	StandardTapSpec("Lectro-v0", true, 0.99, time.Second/10, 512),
	StandardTapSpec("PineapplePen-v0", true, 0.98, time.Second/10, 512),
	StandardTapSpec("SushiNinjaDash-v0", true, 0.98, time.Second/10, 512),
	StandardTapSpec("PopUp-v0", true, 0.98, time.Second/10, 2048),
	Colorize(StandardTapSpec("ColorCircles-v0", true, 0.98, time.Second/10, 512)),
	StandardTapSpec("PanicDrop-v0", true, 0.98, time.Second/10, 512),
	StandardTapSpec("TapTapDash-v0", true, 0.98, time.Second/10, 512),
	WithRewardScale(StandardTapSpec("NinjaRun-v0", true, 0.98, time.Second/10, 512),
		1.0/100),
	WithRewardScale(StandardTapSpec("KibaKumbaShadowRun-v0", true, 0.98,
		time.Second/10, 512), 1.0/250),
	StandardTapSpec("FlappyBird-v0", true, 0.99, time.Second/10, 512),
	StandardTapSpec("StickFreak-v0", false, 0.98, time.Second/10, 512),
	StandardTapSpec("Basketball-v0", false, 0.95, time.Second/10, 512),
	StandardTapSpec("TowerMania-v0", false, 0.99, time.Second/10, 512),
	StandardTapSpec("StackTowerClassic-v0", false, 0.99, time.Second/10, 1024),
	StandardKeySpec("Twins-v0", false, 0.98, time.Second/10, 512),
	StandardKeySpec("RedHead-v0", false, 0.98, time.Second/10, 2048),
	StandardKeySpec("CartoonFlight-v0", false, 0.95, time.Second/8, 512),
	StandardKeySpec("CartoonFlight-v1", false, 0.95, time.Second/8, 512),
	StandardKeySpec("TRex-v0", false, 0.98, time.Second/10, 512),
	StandardKeySpec("Cars-v0", false, 0.98, time.Second/10, 512),
	StandardKeySpec("MeatBoyClicker-v0", false, 0.98, time.Second/10, 512),
	WithRewardScale(StandardKeySpec("DoodleJump-v0", false, 0.98, time.Second/10, 2048),
		1.0/500),
	WithRewardScale(StandardKeySpec("HopDontStop-v0", false, 0.98, time.Second/10, 512),
		1.0/250),
	WithRewardScale(StandardTapSpec("UfoRun-v0", false, 0.99, time.Second/10, 512),
		1.0/100),
	Colorize(StandardKeySpec("ColorTease-v0", true, 0.98, time.Second/10, 1024)),
	StandardMouseSpec("PizzaNinja3-v0", false, 0.99, time.Second/10, 2048),
	StandardMouseSpec("SoccerGirl-v1", false, 0.98, time.Second/10, 512),
	WithRewardScale(StandardMouseSpec("ClickThemAll-v0", false, 0.98, time.Second/10,
		2048), 1.0/1000),
	Colorize(StandardMouseSpec("Colorpop-v0", false, 0.99, time.Second/10, 512)),
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

// MustSpecForName is like SpecForName, but it exits if
// the spec is not found.
func MustSpecForName(name string) *EnvSpec {
	spec := SpecForName(name)
	if spec == nil {
		essentials.Die("unsupported environment:", name)
	}
	return spec
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

		HistorySize: 1,
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

// StandardMouseSpec is like StandardKeySpec, except with
// MouseActor instead of KeyActor.
func StandardMouseSpec(name string, noHold bool, discount float64,
	frameTime time.Duration, batchSize int) *EnvSpec {
	res := StandardKeySpec(name, noHold, discount, frameTime, batchSize)
	res.Wrap = func(e muniverse.Env) muniverse.Env {
		return muniverse.CursorEnv(e, res.Width/2, res.Height/2)
	}
	res.MakeActor = func() Actor {
		return &MouseActor{
			Width:    res.Width,
			Height:   res.Height,
			NoHold:   noHold,
			Discrete: true,
		}
	}
	return res
}

// Colorize changes a standard spec to use color.
func Colorize(e *EnvSpec) *EnvSpec {
	e.Observer.(*DownsampleObserver).Color = true
	return e
}

// WithHistSize changes the history size of a spec.
func WithHistSize(e *EnvSpec, size int) *EnvSpec {
	e.HistorySize = size
	return e
}

// WithRewardScale changes the RewardScale of a spec.
func WithRewardScale(e *EnvSpec, scale float64) *EnvSpec {
	e.RewardScale = scale
	return e
}
