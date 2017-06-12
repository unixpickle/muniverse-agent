package main

import (
	"time"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
)

// An Env is an anyrl.Env which wraps a muniverse.Env.
type Env struct {
	Creator  anyvec.Creator
	RawEnv   muniverse.Env
	Actor    Actor
	Observer Observer

	FrameTime time.Duration
	MaxSteps  int

	timestep  int
	lastFrame anyvec.Vector
}

// Reset resets the environment.
func (e *Env) Reset() (obs anyvec.Vector, err error) {
	defer essentials.AddCtxTo("reset", &err)

	e.Actor.Reset()
	e.timestep = 0

	err = e.RawEnv.Reset()
	if err != nil {
		return
	}

	rawObs, err := e.RawEnv.Observe()
	if err != nil {
		return
	}
	obsVec, err := e.Observer.ObsVec(e.Creator, rawObs)
	if err != nil {
		return
	}
	e.lastFrame = obsVec.Copy()
	obs = joinFrames(obsVec, obsVec)

	return
}

// Step takes a step in the environment.
func (e *Env) Step(action anyvec.Vector) (obs anyvec.Vector, reward float64,
	done bool, err error) {
	events := e.Actor.Events(action)
	reward, done, err = e.RawEnv.Step(e.FrameTime, events...)
	if err != nil {
		return
	}

	rawObs, err := e.RawEnv.Observe()
	if err != nil {
		return
	}
	obsVec, err := e.Observer.ObsVec(e.Creator, rawObs)
	if err != nil {
		return
	}
	obs = joinFrames(e.lastFrame, obsVec)
	e.lastFrame = obsVec.Copy()

	e.timestep++
	if e.timestep >= e.MaxSteps {
		done = true
	}

	return
}

func joinFrames(f1, f2 anyvec.Vector) anyvec.Vector {
	joined := f1.Creator().Concat(f1, f2)
	res := f1.Creator().MakeVector(f1.Len() * 2)
	anyvec.Transpose(joined, res, 2)
	return res
}
