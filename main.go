package main

import (
	"flag"
	"log"
	"time"

	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"

	_ "github.com/unixpickle/anyplugin"
)

// Flags contains the command-line options.
type Flags struct {
	EnvName     string
	OutFile     string
	CriticFile  string
	NumParallel int
	MaxSteps    int
	RecordDir   string

	A3C           bool
	A3CEntropyReg float64
	A3CStep       float64
	A3CInterval   int
	A3CSaveTime   time.Duration

	DemosDir       string
	DemoBatch      int
	DemoValidation string
	DemoL2Reg      float64

	ImageName   string
	GamesDir    string
	Compression int
}

func main() {
	creator := anyvec32.CurrentCreator()

	var flags Flags
	flag.StringVar(&flags.EnvName, "env", "", "muniverse environment name")
	flag.StringVar(&flags.OutFile, "out", "trained_policy", "policy output file")
	flag.StringVar(&flags.CriticFile, "critic", "trained_critic", "A3C critic output file")
	flag.IntVar(&flags.NumParallel, "numparallel", 8, "parallel environments")
	flag.IntVar(&flags.MaxSteps, "maxsteps", 600, "max time steps per episode")
	flag.StringVar(&flags.RecordDir, "record", "", "directory to store recordings")
	flag.BoolVar(&flags.A3C, "a3c", false, "use A3C instead of TRPO")
	flag.Float64Var(&flags.A3CEntropyReg, "a3creg", 0.01, "A3C entropy regularization")
	flag.Float64Var(&flags.A3CStep, "a3cstep", 1e-5, "A3C step size")
	flag.IntVar(&flags.A3CInterval, "a3cinterval", 20, "A3C frames per update")
	flag.DurationVar(&flags.A3CSaveTime, "a3csave", time.Minute*5, "A3C save interval")
	flag.StringVar(&flags.DemosDir, "demos", "", "supervised demonstrations to train with")
	flag.StringVar(&flags.DemoValidation, "demovalidation", "", "validation demonstrations")
	flag.IntVar(&flags.DemoBatch, "demobatch", 16, "batch size (supervised only)")
	flag.Float64Var(&flags.DemoL2Reg, "demol2reg", 0, "L2 regularization (supervised only)")
	flag.StringVar(&flags.ImageName, "image", "", "custom Docker image")
	flag.StringVar(&flags.GamesDir, "gamesdir", "", "custom games directory")
	flag.IntVar(&flags.Compression, "compression", -1, "screen image compression (0-100)")
	flag.Parse()

	if flags.EnvName == "" {
		essentials.Die("Missing -env flag. See -help for more.")
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
	} else if flags.A3C {
		var critic anyrnn.Block
		if err := serializer.LoadAny(flags.CriticFile, &critic); err != nil {
			log.Println("Creating new critic...")
			critic = MakeCritic(creator)
		} else {
			log.Println("Loaded critic.")
		}
		A3C(flags, spec, policy, critic)
	} else {
		TRPO(flags, spec, policy)
	}
}
