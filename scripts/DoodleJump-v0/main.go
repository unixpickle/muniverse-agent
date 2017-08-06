package main

import (
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/unixpickle/muniverse"
	"github.com/unixpickle/muniverse/chrome"
)

const (
	CharacterHeight = 50
	PlatformPixels  = 3000
)

var Spec = muniverse.SpecForName("DoodleJump-v0")

func main() {
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		go record()
	}
	select {}
}

func record() {
	env, _ := muniverse.NewEnv(Spec)
	env = muniverse.RecordEnv(env, "doodle_demos")
	defer env.Close()
	for {
		env.Reset()
		direction := 0
		lastObs, _ := env.Observe()
		for i := 0; true; i++ {
			var actions []interface{}

			// Only switch directions periodically.
			if rand.Intn(3) == 0 {
				rgb, _, _, _ := muniverse.RGB(lastObs)
				jumperX, jumperY := jumperPosition(rgb)

				leftPlatforms := platformsInRegion(rgb, 0, jumperY, jumperX-50)
				middlePlatforms := platformsInRegion(rgb, jumperX-50, jumperY, 100)
				rightPlatforms := platformsInRegion(rgb, jumperX+50, jumperY,
					Spec.Width-(jumperX+50))

				probs := softmax(leftPlatforms, middlePlatforms, rightPlatforms)

				r := rand.Float64()
				var newDirection int
				if r < probs[0] {
					newDirection = -1
				} else if r < probs[0]+probs[1] {
					newDirection = 0
				} else {
					newDirection = 1
				}
				actions = directionChangeActions(direction, newDirection)
				direction = newDirection
			}

			_, done, _ := env.Step(time.Second/10, actions...)
			lastObs, _ = env.Observe()
			if done {
				break
			}
		}
	}
}

func jumperPosition(rgb []uint8) (x, y int) {
	for i := 0; i < len(rgb); i += 3 {
		r, g, b := rgb[i], rgb[i+1], rgb[i+2]
		dist := abs(int(r)-0xcb) + abs(int(g)-0xc9) + abs(int(b)-0x16)
		if dist < 5 {
			x = (i % (3 * Spec.Width)) / 3
			y = i / (3 * Spec.Width)

			y += CharacterHeight
			if y >= Spec.Height {
				y = Spec.Height - 1
			}

			return
		}
	}
	return
}

func platformsInRegion(rgb []uint8, topX, topY, width int) float64 {
	if topY >= Spec.Height || topX >= Spec.Width || width <= 0 {
		return 0
	}
	if topY < 0 {
		topY = 0
	}
	if topX < 0 {
		width += topX
		topX = 0
	}
	var sum float64
	for y := topY; y < Spec.Height; y++ {
		for x := topX; x < topX+width && x < Spec.Width; x++ {
			idx := 3 * (Spec.Width*y + x)
			red := rgb[idx]
			if red < 0xe0 {
				sum++
			}
		}
	}
	return sum / PlatformPixels
}

func softmax(values ...float64) []float64 {
	var res []float64
	var sum float64
	for _, x := range values {
		y := math.Exp(x)
		res = append(res, y)
		sum += y
	}
	for i, x := range res {
		res[i] = x / sum
	}
	return res
}

func directionChangeActions(oldDir, newDir int) []interface{} {
	var actions []interface{}
	if newDir != oldDir {
		if oldDir != 0 {
			evt := directionEvent(oldDir)
			evt.Type = chrome.KeyUp
			actions = append(actions, &evt)
		}
		if newDir != 0 {
			evt := directionEvent(newDir)
			evt.Type = chrome.KeyDown
			actions = append(actions, &evt)
		}
	}
	return actions
}

func directionEvent(direction int) chrome.KeyEvent {
	if direction == -1 {
		return chrome.KeyEvents["ArrowLeft"]
	} else if direction == 1 {
		return chrome.KeyEvents["ArrowRight"]
	} else {
		panic("bad direction")
	}
}

func abs(i int) int {
	if i < 0 {
		return -i
	}
	return i
}
