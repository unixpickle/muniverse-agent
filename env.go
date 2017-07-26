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

	timestep int
	joiner   *ObsJoiner
}

// NewEnv creates an environment according to the flags
// and specification.
//
// It is the caller's responsibility to close RawEnv once
// it is done using the environment.
func NewEnv(c anyvec.Creator, flags *TrainingFlags, spec *EnvSpec) *Env {
	opts := &muniverse.Options{}
	if flags.ImageName != "" {
		opts.CustomImage = flags.ImageName
	}
	if flags.GamesDir != "" {
		opts.GamesDir = flags.GamesDir
	}
	if flags.Compression >= 0 {
		if flags.Compression > 100 {
			essentials.Die("invalid compression level:", flags.Compression)
		}
		opts.Compression = true
		opts.CompressionQuality = flags.Compression
	}
	env, err := muniverse.NewEnvOptions(spec.EnvSpec, opts)
	if err != nil {
		essentials.Die(err)
	}
	if spec.Wrap != nil {
		env = spec.Wrap(env)
	}
	if flags.RecordDir != "" {
		env = muniverse.RecordEnv(env, flags.RecordDir)
	}
	return &Env{
		Creator:   c,
		RawEnv:    env,
		Actor:     spec.MakeActor(),
		Observer:  spec.Observer,
		FrameTime: spec.FrameTime,
		MaxSteps:  flags.MaxSteps,
		joiner:    &ObsJoiner{HistorySize: spec.HistorySize},
	}
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
	e.joiner.Reset(obsVec)
	obs = e.joiner.Step(obsVec)

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
	obs = e.joiner.Step(obsVec)

	e.timestep++
	if e.timestep >= e.MaxSteps {
		done = true
	}

	return
}
