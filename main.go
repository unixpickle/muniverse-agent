package main

import (
	"compress/flate"
	"flag"
	"log"
	"math"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"

	_ "github.com/unixpickle/anyplugin"
)

// Flags contains the command-line options.
type Flags struct {
	EnvName     string
	OutFile     string
	NumParallel int
	MaxSteps    int
	RecordDir   string

	DemosDir       string
	DemoBatch      int
	DemoValidation string

	ImageName string
	GamesDir  string
}

func main() {
	creator := anyvec32.CurrentCreator()

	var flags Flags
	flag.StringVar(&flags.EnvName, "env", "", "muniverse environment name")
	flag.StringVar(&flags.OutFile, "out", "trained_policy", "policy output file")
	flag.IntVar(&flags.NumParallel, "numparallel", 8, "parallel environments")
	flag.IntVar(&flags.MaxSteps, "maxsteps", 600, "max time steps per episode")
	flag.StringVar(&flags.RecordDir, "record", "", "directory to store recordings")
	flag.StringVar(&flags.DemosDir, "demos", "", "supervised demonstrations to train with")
	flag.StringVar(&flags.DemoValidation, "demovalidation", "", "validation demonstrations")
	flag.IntVar(&flags.DemoBatch, "demobatch", 16, "batch size (supervised only)")
	flag.StringVar(&flags.ImageName, "image", "", "custom Docker image")
	flag.StringVar(&flags.GamesDir, "gamesdir", "", "custom games directory")
	flag.Parse()

	if flags.EnvName == "" {
		essentials.Die("Missing -env flag. See -help for more.")
	} else if flags.ImageName != "" && flags.GamesDir != "" {
		essentials.Die("Cannot use -image and -gamesdir together.")
	}

	spec := SpecForName(flags.EnvName)
	if spec == nil {
		essentials.Die("unsupported environment:", flags.EnvName)
	}

	var policy anyrnn.Block
	if err := serializer.LoadAny(flags.OutFile, &policy); err != nil {
		log.Println("Creating new policy...")
		policy = MakePolicy(creator, spec)
	} else {
		log.Println("Loaded policy.")
	}

	if flags.DemosDir != "" {
		SupervisedTrain(flags, spec, policy)
		return
	}

	actionSpace := spec.MakeActor().ActionSpace()

	roller := &anyrl.RNNRoller{
		Block:       policy,
		ActionSpace: actionSpace,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	trpo := &anypg.TRPO{
		NaturalPG: anypg.NaturalPG{
			Policy:      policy,
			Params:      anynet.AllParameters(policy),
			ActionSpace: actionSpace,

			// Speed things up a bit.
			Reduce: (&anyrl.FracReducer{
				Frac:          0.1,
				MakeInputTape: roller.MakeInputTape,
			}).Reduce,

			ApplyPolicy:  ApplyPolicy,
			ActionJudger: &anypg.QJudger{Discount: spec.DiscountFactor},
		},
		LogLineSearch: func(kl, improvement anyvec.Numeric) {
			log.Printf("line search: kl=%f improvement=%f", kl, improvement)
		},
	}

	// Train on a background goroutine so that we can
	// listen for Ctrl+C on the main goroutine.
	var trainLock sync.Mutex
	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			// Join the rollouts into one set.
			rollouts := gatherRollouts(flags, spec, roller)
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f stddev=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()))

			// Train on the rollouts.
			log.Println("Training on batch...")
			grad := trpo.Run(r)
			grad.AddToVars()
			trainLock.Lock()
			if err := serializer.SaveAny(flags.OutFile, policy); err != nil {
				essentials.Die("save policy:", err)
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

func gatherRollouts(flags Flags, spec *EnvSpec,
	roller *anyrl.RNNRoller) []*anyrl.RolloutSet {
	resChan := make(chan *anyrl.RolloutSet, spec.BatchSize)

	requests := make(chan struct{}, spec.BatchSize)
	for i := 0; i < spec.BatchSize; i++ {
		requests <- struct{}{}
	}
	close(requests)

	var wg sync.WaitGroup
	for i := 0; i < flags.NumParallel; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var env muniverse.Env
			var err error
			if flags.ImageName != "" {
				env, err = muniverse.NewEnvContainer(flags.ImageName, spec.EnvSpec)
			} else if flags.GamesDir != "" {
				env, err = muniverse.NewEnvGamesDir(flags.GamesDir, spec.EnvSpec)
			} else {
				env, err = muniverse.NewEnv(spec.EnvSpec)
			}
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

			rlEnv := &Env{
				Creator:   anynet.AllParameters(roller.Block)[0].Vector.Creator(),
				RawEnv:    env,
				Actor:     spec.MakeActor(),
				Observer:  spec.Observer,
				FrameTime: spec.FrameTime,
				MaxSteps:  flags.MaxSteps,
			}
			for _ = range requests {
				rollout, err := roller.Rollout(rlEnv)
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

	logInterval := essentials.MaxInt(1, spec.BatchSize/32)

	var res []*anyrl.RolloutSet
	var batchRewardSum float64
	var numBatchReward int
	for item := range resChan {
		res = append(res, item)
		numBatchReward++
		batchRewardSum += item.Rewards.Mean()
		if numBatchReward == logInterval || len(res) == spec.BatchSize {
			log.Printf("sub_mean=%f", batchRewardSum/float64(numBatchReward))
			numBatchReward = 0
			batchRewardSum = 0
		}
	}
	return res
}
