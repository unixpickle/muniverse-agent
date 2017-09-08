package main

import (
	"compress/flate"
	"flag"
	"log"
	"math"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func TRPO(c anyvec.Creator, args []string) {
	fs := flag.NewFlagSet("trpo", flag.ExitOnError)
	flags := &TrainingFlags{}
	flags.Add(fs)
	fs.Parse(args)

	spec := MustSpecForName(flags.EnvName)
	policy, _ := LoadOrMakeAgent(c, spec, flags.PolicyFile, "", false)

	actionSpace := spec.MakeActor().ActionSpace()

	roller := &anyrl.RNNRoller{
		Block:       policy,
		ActionSpace: actionSpace,

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func(c anyvec.Creator) (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(c, flate.DefaultCompression)
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

			ApplyPolicy:  ApplyBlock,
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
			rollouts := gatherTRPORollouts(flags, spec, roller)
			r := anyrl.PackRolloutSets(c, rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f stddev=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()))

			// Train on the rollouts.
			log.Println("Training on batch...")
			grad := trpo.Run(r)
			grad.AddToVars()
			trainLock.Lock()
			if err := serializer.SaveAny(flags.PolicyFile, policy); err != nil {
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

func gatherTRPORollouts(flags *TrainingFlags, spec *EnvSpec,
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
			env := NewEnv(flags, spec)
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
