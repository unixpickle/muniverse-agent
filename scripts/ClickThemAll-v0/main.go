package main

import (
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/unixpickle/essentials"
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
	spec := muniverse.SpecForName("ClickThemAll-v0")
	env, _ := muniverse.NewEnv(spec)
	env = muniverse.CursorEnv(env, spec.Width/2, spec.Height/2)
	env = muniverse.RecordEnv(env, "click_demos")
	defer env.Close()
	for {
		env.Reset()
		lastObs, _ := env.Observe()
		mouseX := spec.Width / 2
		mouseY := spec.Height / 2
		for i := 0; true; i++ {
			randomMovement(spec, &mouseX, &mouseY)
			actions := []interface{}{
				&chrome.MouseEvent{
					Type: chrome.MouseMoved,
					X:    mouseX,
					Y:    mouseY,
				},
			}

			if shouldClick(spec, mouseX, mouseY, lastObs) {
				click := chrome.MouseEvent{
					Type:       chrome.MousePressed,
					X:          mouseX,
					Y:          mouseY,
					Button:     chrome.LeftButton,
					ClickCount: 1,
				}
				unclick := click
				unclick.Type = chrome.MouseReleased
				actions = append(actions, &click, unclick)
			}

			_, done, _ := env.Step(time.Second/10, actions...)
			lastObs, _ = env.Observe()
			if done {
				break
			}
		}
	}
}

func randomMovement(spec *muniverse.EnvSpec, mouseX, mouseY *int) {
	opts := mouseOptions()
	opt := opts[rand.Intn(len(opts))]
	*mouseX += opt[0]
	*mouseY += opt[1]
	*mouseX = essentials.MaxInt(0, essentials.MinInt(*mouseX, spec.Width-1))
	*mouseY = essentials.MaxInt(0, essentials.MinInt(*mouseY, spec.Height-1))
}

func shouldClick(spec *muniverse.EnvSpec, mouseX, mouseY int,
	obs muniverse.Obs) bool {
	buf, _, _, _ := muniverse.RGB(obs)
	idx := 3 * (mouseX + mouseY*spec.Width)
	color := 0
	for _, c := range buf[idx : idx+3] {
		color += int(c)
	}

	// Favor colored pixels for clicks.
	return (rand.Intn(10) < 8 && color > 0x80 && color < 0xff*3-0x80) ||
		rand.Intn(10) < 2
}

func mouseOptions() [][2]int {
	res := make([][2]int, 1, 3*5+1)
	res[0] = [2]int{0, 0}
	for _, radius := range []float64{10, 40, 80} {
		for i := 0; i < 5; i++ {
			angle := math.Pi * 2 * float64(i) / 5
			x := math.Cos(angle) * radius
			y := math.Sin(angle) * radius
			res = append(res, [2]int{int(x), int(y)})
		}
	}
	return res
}
