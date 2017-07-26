package main

import (
	"flag"
	"log"
	"time"

	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

// A3CFlags are the flags for A3C.
type A3CFlags struct {
	TrainingFlags

	CriticFile string
	EntropyReg float64
	Step       float64
	Interval   int
	SaveTime   time.Duration
}

// Add adds the flags to the flag set.
func (a *A3CFlags) Add(fs *flag.FlagSet) {
	a.TrainingFlags.Add(fs)
	fs.StringVar(&a.CriticFile, "critic", "trained_critic", "filename for critic network")
	fs.Float64Var(&a.EntropyReg, "reg", 0.01, "A3C entropy regularization")
	fs.Float64Var(&a.Step, "step", 1e-5, "A3C step size")
	fs.IntVar(&a.Interval, "interval", 20, "A3C frames per update")
	fs.DurationVar(&a.SaveTime, "save", time.Minute*5, "A3C save interval")
}

func A3C(c anyvec.Creator, args []string) {
	fs := flag.NewFlagSet("a3c", flag.ExitOnError)
	flags := &A3CFlags{}
	flags.Add(fs)
	fs.Parse(args)

	spec := MustSpecForName(flags.EnvName)
	policy, critic := LoadOrMakeAgent(c, spec, flags.PolicyFile, flags.CriticFile, true)
	agent := MakeAgent(c, spec, policy, critic)

	log.Println("Initializing environments...")
	var environments []anyrl.Env
	for i := 0; i < flags.NumParallel; i++ {
		e := NewEnv(c, &flags.TrainingFlags, spec)
		defer e.RawEnv.Close()
		environments = append(environments, e)
	}

	paramServer := anya3c.RMSPropParamServer(agent, agent.AllParameters(),
		flags.Step, anysgd.RMSProp{DecayRate: 0.99})
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
		MaxSteps: flags.Interval,
		Regularizer: &anypg.EntropyReg{
			Entropyer: agent.ActionSpace.(anyrl.Entropyer),
			Coeff:     flags.EntropyReg,
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
			case <-time.After(flags.SaveTime):
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

func saveA3C(flags *A3CFlags, paramServer anya3c.ParamServer) error {
	agent, err := paramServer.LocalCopy()
	if err != nil {
		return err
	}
	newPolicy, newCritic := DecomposeAgent(agent.Agent)
	if err := serializer.SaveAny(flags.PolicyFile, newPolicy); err != nil {
		return err
	}
	return serializer.SaveAny(flags.CriticFile, newCritic)
}
