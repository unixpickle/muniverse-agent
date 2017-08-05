package main

import (
	"math/rand"
	"runtime"
	"time"

	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

func main() {
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		go record()
	}
	select {}
}

func record() {
	spec := muniverse.SpecForName("FlappyBird-v0")
	env, _ := muniverse.NewEnv(spec)
	env = muniverse.RecordEnv(env, "flappy_demos")
	defer env.Close()
	for {
		env.Reset()
		env.Observe()
		untilNext := rand.Intn(4) + 9
		for i := 0; true; i++ {
			var actions []interface{}
			untilNext--
			if untilNext == 0 {
				untilNext = rand.Intn(4) + 9
				click := chrome.MouseEvent{
					Type:       chrome.MousePressed,
					X:          100,
					Y:          100,
					Button:     chrome.LeftButton,
					ClickCount: 1,
				}
				unclick := click
				unclick.Type = chrome.MouseReleased
				actions = []interface{}{&click, &unclick}
			}
			_, done, _ := env.Step(time.Second/10, actions...)
			env.Observe()
			if done {
				break
			}
		}
	}
}
