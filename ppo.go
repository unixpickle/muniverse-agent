package main

import (
	"compress/flate"
	"flag"
	"log"
	"math"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

type PPOFlags struct {
	TrainingFlags

	CriticFile   string
	Lambda       float64
	Epsilon      float64
	RegCoeff     float64
	KLReg        bool
	CriticWeight float64
	Step         float64
	Epochs       int
	BatchSteps   int
}

func (p *PPOFlags) Add(fs *flag.FlagSet) {
	p.TrainingFlags.Add(fs)
	fs.StringVar(&p.CriticFile, "critic", "trained_critic", "filename for critic network")
	fs.Float64Var(&p.Lambda, "lambda", 0.95, "GAE coefficient")
	fs.Float64Var(&p.Epsilon, "epsilon", 0.1, "PPO probability epsilon")
	fs.Float64Var(&p.RegCoeff, "reg", 0.01, "regularization strength")
	fs.BoolVar(&p.KLReg, "klreg", false, "use KL regularization (instead of entropy)")
	fs.Float64Var(&p.CriticWeight, "criticweight", 1, "importance of critic gradient")
	fs.Float64Var(&p.Step, "step", 3e-4, "SGD step size (with Adam)")
	fs.IntVar(&p.Epochs, "epochs", 10, "SGD epochs per batch")
	fs.IntVar(&p.BatchSteps, "batchsteps", 2048, "minimum steps per batch")
}

func PPO(c anyvec.Creator, args []string) {
	fs := flag.NewFlagSet("ppo", flag.ExitOnError)
	flags := &PPOFlags{}
	flags.Add(fs)
	fs.Parse(args)

	spec := MustSpecForName(flags.EnvName)
	policy, critic := LoadOrMakeAgent(c, spec, flags.PolicyFile, flags.CriticFile, true)
	agent := MakeAgent(c, spec, policy, critic)

	roller := &anyrl.RNNRoller{
		Block:       policy,
		ActionSpace: agent.ActionSpace,

		// Compress the input frames as we store them.
		// This may not be necessary with relatively small
		// batch sizes.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	ppo := &anypg.PPO{
		Params: agent.AllParameters(),
		Base: func(in lazyseq.Rereader) lazyseq.Rereader {
			return ApplyBlock(in, agent.Base)
		},
		Actor: func(in lazyseq.Rereader) lazyseq.Rereader {
			return ApplyBlock(in, agent.Actor)
		},
		Critic: func(in lazyseq.Rereader) lazyseq.Rereader {
			return ApplyBlock(in, agent.Critic)
		},
		ActionSpace:  agent.ActionSpace,
		CriticWeight: flags.CriticWeight,
		Discount:     spec.DiscountFactor,
		Lambda:       flags.Lambda,
		Epsilon:      flags.Epsilon,
	}
	if flags.KLReg {
		ppo.Regularizer = &anypg.KLReg{
			Coeff: flags.RegCoeff,
			Base:  c.MakeVector(spec.MakeActor().ParamLen()),
			KLer:  agent.ActionSpace.(anyrl.KLer),
		}
	} else {
		ppo.Regularizer = &anypg.EntropyReg{
			Coeff:     flags.RegCoeff,
			Entropyer: agent.ActionSpace.(anyrl.Entropyer),
		}
	}

	// Train on a background goroutine so that we can
	// listen for Ctrl+C on the main goroutine.
	var trainLock sync.Mutex
	var transformer anysgd.Adam
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			// Join the rollouts into one set.
			rollouts := gatherPPORollouts(flags, spec, roller)
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f count=%d error_margin=%f", batchIdx,
				r.Rewards.Mean(), len(r.Rewards),
				math.Sqrt(r.Rewards.Variance()/float64(len(r.Rewards))))

			// Train on the rollouts.
			log.Println("Training on batch...")
			adv := ppo.Advantage(r)
			for i := 0; i < flags.Epochs; i++ {
				g, terms := ppo.Run(r, adv)
				g = transformer.Transform(g)
				g.Scale(c.MakeNumeric(flags.Step))
				g.AddToVars()
				log.Printf("iteration %d: actor=%f critic=%f reg=%f", i,
					terms.MeanAdvantage, terms.MeanCritic,
					terms.MeanRegularization)
			}

			trainLock.Lock()
			if err := serializer.SaveAny(flags.PolicyFile, policy); err != nil {
				essentials.Die("save policy:", err)
			}
			if err := serializer.SaveAny(flags.CriticFile, critic); err != nil {
				essentials.Die("save critic:", err)
			}
			trainLock.Unlock()
		}
	}()

	log.Println("Running. Press Ctrl+C to stop.")
	<-rip.NewRIP().Chan()

	// Avoid the race condition where we save during
	// exit.
	trainLock.Lock()
}

func gatherPPORollouts(flags *PPOFlags, spec *EnvSpec,
	roller *anyrl.RNNRoller) []*anyrl.RolloutSet {
	resChan := make(chan *anyrl.RolloutSet, spec.BatchSize)

	requests := make(chan struct{}, flags.NumParallel)
	for i := 0; i < flags.NumParallel; i++ {
		requests <- struct{}{}
	}

	var wg sync.WaitGroup
	for i := 0; i < flags.NumParallel; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			creator := anynet.AllParameters(roller.Block)[0].Vector.Creator()
			env := NewEnv(creator, &flags.TrainingFlags, spec)
			defer env.RawEnv.Close()
			for _ = range requests {
				rollout, err := roller.Rollout(env)
				if err != nil {
					essentials.Die("rollout error:", err)
				}
				resChan <- rollout
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resChan)
	}()

	var res []*anyrl.RolloutSet
	var numSteps int
	var closed bool
	for item := range resChan {
		res = append(res, item)
		numSteps += len(item.Rewards[0])
		if numSteps < flags.BatchSteps {
			requests <- struct{}{}
		} else if !closed {
			closed = true
			close(requests)
		}
	}
	return res
}
