package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/unixpickle/anyvec/anyvec32"

	_ "github.com/unixpickle/anyplugin"
)

type GeneralFlags struct {
	EnvName    string
	PolicyFile string
}

func (g *GeneralFlags) Add(fs *flag.FlagSet) {
	fs.StringVar(&g.EnvName, "env", "", "muniverse environment name")
	fs.StringVar(&g.PolicyFile, "out", "trained_policy", "filename for policy network")
}

type TrainingFlags struct {
	GeneralFlags

	MaxSteps    int
	NumParallel int
	RecordDir   string
	ImageName   string
	GamesDir    string
	Compression int
}

func (t *TrainingFlags) Add(fs *flag.FlagSet) {
	t.GeneralFlags.Add(fs)
	fs.IntVar(&t.NumParallel, "numparallel", 8, "parallel environments")
	fs.IntVar(&t.MaxSteps, "maxsteps", 600, "max time steps per episode")
	fs.StringVar(&t.RecordDir, "record", "", "directory to store recordings")
	fs.StringVar(&t.ImageName, "image", "", "custom Docker image")
	fs.StringVar(&t.GamesDir, "gamesdir", "", "custom games directory")
	fs.IntVar(&t.Compression, "compression", -1, "screen image compression (0-100)")
}

func main() {
	creator := anyvec32.CurrentCreator()

	if len(os.Args) < 2 || os.Args[1] == "-help" {
		dieUsage()
	}

	log.Println("Running with arguments:", os.Args[1:])

	switch os.Args[1] {
	case "trpo":
		TRPO(creator, os.Args[2:])
	case "ppo":
		PPO(creator, os.Args[2:])
	case "a3c":
		A3C(creator, os.Args[2:])
	case "clone":
		Clone(creator, os.Args[2:])
	default:
		fmt.Fprintln(os.Stderr, "Unknown sub-command:", os.Args[1])
	}
}

func dieUsage() {
	lines := []string{
		"Usage: muniverse-agent <sub-command> [args | -help]",
		"",
		"Available sub-commands:",
		" trpo      train a policy with TRPO",
		" ppo       train a policy with PPO",
		" a3c       train a policy and critic with A3C",
		" clone     clone a policy from demonstrations",
	}
	for _, line := range lines {
		fmt.Fprintln(os.Stderr, line)
	}
	os.Exit(1)
}
