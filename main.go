package main

import (
	"flag"
	"log"
	"time"

	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anya3c"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

const SaveInterval = time.Minute * 5

// Flags contains the command-line options.
type Flags struct {
	EnvName    string
	OutFile    string
	NumWorkers int
	MaxSteps   int
	RecordDir  string

	DemosDir       string
	DemoBatch      int
	DemoValidation string
}

func main() {
	creator := anyvec32.CurrentCreator()

	var flags Flags
	flag.StringVar(&flags.EnvName, "env", "", "muniverse environment name")
	flag.StringVar(&flags.OutFile, "out", "trained_agent", "agent output file")
	flag.IntVar(&flags.NumWorkers, "workers", 8, "number of A3C workers")
	flag.IntVar(&flags.MaxSteps, "maxsteps", 600, "max time steps per episode")
	flag.StringVar(&flags.RecordDir, "record", "", "directory to store recordings")
	flag.StringVar(&flags.DemosDir, "demos", "", "supervised demonstrations to train with")
	flag.StringVar(&flags.DemoValidation, "demovalidation", "", "validation demonstrations")
	flag.IntVar(&flags.DemoBatch, "demobatch", 16, "batch size (supervised only)")
	flag.Parse()

	if flags.EnvName == "" {
		essentials.Die("Missing -env flag. See -help for more.")
	}

	spec := SpecForName(flags.EnvName)
	if spec == nil {
		essentials.Die("unsupported environment:", flags.EnvName)
	}

	agent := &anya3c.Agent{ActionSpace: spec.MakeActor().ActionSpace()}
	err := serializer.LoadAny(flags.OutFile, &agent.Base, &agent.Actor, &agent.Critic)
	if err != nil {
		log.Println("Creating new agent...")
		agent = MakeAgent(creator, spec)
	} else {
		log.Println("Loaded agent.")
	}

	if flags.DemosDir != "" {
		SupervisedTrain(flags, spec, anyrnn.Stack{agent.Base, agent.Actor})
	} else {
		RLTrain(flags, spec, agent)
	}
	serializer.SaveAny(flags.OutFile, agent.Base, agent.Actor, agent.Critic)
}

func RLTrain(flags Flags, spec *EnvSpec, agent *anya3c.Agent) {
	creator := agent.AllParameters()[0].Vector.Creator()

	log.Println("Initializing environments...")
	var environments []anyrl.Env
	for i := 0; i < flags.NumWorkers; i++ {
		env, err := muniverse.NewEnv(spec.EnvSpec)
		if err != nil {
			essentials.Die("create environment:", err)
		}
		if spec.Wrap != nil {
			env = spec.Wrap(env)
		}
		if flags.RecordDir != "" {
			env = muniverse.RecordEnv(env, flags.RecordDir)
		}
		defer env.Close()

		environments = append(environments, &Env{
			Creator:   creator,
			RawEnv:    env,
			Actor:     spec.MakeActor(),
			Observer:  spec.Observer,
			FrameTime: spec.FrameTime,
			MaxSteps:  flags.MaxSteps,
		})
	}

	paramServer := anya3c.RMSPropParamServer(agent, agent.AllParameters(),
		1e-4, anysgd.RMSProp{DecayRate: 0.99})
	defer paramServer.Close()

	a3c := &anya3c.A3C{
		ParamServer: paramServer,
		Logger: &anya3c.AvgLogger{
			Creator: creator,
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
		MaxSteps: 5,
		Regularizer: &anypg.EntropyReg{
			Entropyer: agent.ActionSpace.(anyrl.Entropyer),
			Coeff:     0.003,
		},
	}

	trainEnd := rip.NewRIP()

	saveDone := make(chan struct{}, 1)
	go func() {
		defer close(saveDone)
		for !trainEnd.Done() {
			agent, err := paramServer.LocalCopy()
			if err != nil {
				essentials.Die(err)
			}
			err = serializer.SaveAny(flags.OutFile, agent.Base,
				agent.Actor, agent.Critic)
			if err != nil {
				essentials.Die(err)
			}
			select {
			case <-time.After(SaveInterval):
			case <-trainEnd.Chan():
			}
		}
	}()

	log.Println("Running A3C...")
	a3c.Run(environments, trainEnd.Chan())

	log.Println("Waiting for network to save...")
	<-saveDone
}
