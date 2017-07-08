package main

import (
	"log"
	"time"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func A3C(flags Flags, spec *EnvSpec, policy, critic anyrnn.Block) {
	c := anynet.AllParameters(policy)[0].Vector.Creator()
	agent := MakeAgent(c, spec, policy, critic)

	log.Println("Initializing environments...")
	var environments []anyrl.Env
	for i := 0; i < flags.NumParallel; i++ {
		e := NewEnv(c, flags, spec)
		defer e.RawEnv.Close()
		environments = append(environments, e)
	}

	paramServer := anya3c.RMSPropParamServer(agent, agent.AllParameters(),
		flags.A3CStep, anysgd.RMSProp{DecayRate: 0.99})
	defer paramServer.Close()

	a3c := &anya3c.A3C{
		ParamServer: paramServer,
		Logger: &anya3c.AvgLogger{
			Creator: c,
			Logger: &anya3c.StandardLogger{
				Episode:    true,
				Update:     true,
				Regularize: true,
			},
			// Only log updates and entropy periodically.
			Update:     60,
			Regularize: 120,
		},
		Discount: spec.DiscountFactor,
		MaxSteps: flags.A3CInterval,
		Regularizer: &anypg.EntropyReg{
			Entropyer: agent.ActionSpace.(anyrl.Entropyer),
			Coeff:     flags.A3CEntropyReg,
		},
	}

	trainEnd := rip.NewRIP()

	saveDone := make(chan struct{}, 1)
	go func() {
		defer close(saveDone)
		for !trainEnd.Done() {
			if err := saveA3C(flags, paramServer); err != nil {
				essentials.Die(err)
			}
			select {
			case <-time.After(flags.A3CSaveTime):
			case <-trainEnd.Chan():
				if err := saveA3C(flags, paramServer); err != nil {
					essentials.Die(err)
				}
			}
		}
	}()

	log.Println("Running A3C...")
	a3c.Run(environments, trainEnd.Chan())

	log.Println("Waiting for network to save...")
	<-saveDone
}

func saveA3C(flags Flags, paramServer anya3c.ParamServer) error {
	agent, err := paramServer.LocalCopy()
	if err != nil {
		return err
	}
	newPolicy, newCritic := DecomposeAgent(agent.Agent)
	if err := serializer.SaveAny(flags.OutFile, newPolicy); err != nil {
		return err
	}
	return serializer.SaveAny(flags.CriticFile, newCritic)
}
