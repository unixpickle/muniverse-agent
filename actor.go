package main

import (
	"math"

	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/muniverse/chrome"
)

// An ActionSpace is a parametric distribution over
// actions which an agent can take.
type ActionSpace interface {
	anyrl.Sampler
	anyrl.LogProber
	anyrl.KLer
}

// An Actor converts action vectors into muniverse events.
//
// An Actor may be stateful, meaning that previous calls
// to Events() may affect later Events() calls.
// This state is reset via Reset().
//
// Actors are not thread-safe.
type Actor interface {
	// ActionSpace returns the space used for sampling.
	ActionSpace() ActionSpace

	// ParamLen returns the input length for parameter
	// vectors of ActionSpace.
	ParamLen() int

	// Reset resets the state of the Actor.
	// This must be called at least once before
	// Events can be called.
	Reset()

	// Events converts a sampled action vector to
	// muniverse events.
	Events(actVec []float64) []interface{}

	// Vectorize performs the pseudo-inverse of Events.
	// If no action vector could produce the events, a
	// reasonable vector should be chosen which would
	// produce some of the events.
	Vectorize(events []interface{}) []float64
}

// KeyActor is an Actor which produces keyboard events
// where each key is controlled by another vector
// component.
type KeyActor struct {
	Keys []string

	// NoHold, if true, indicates that key presses are
	// instantaneous; keys cannot be held down.
	NoHold bool

	// OneHot, if true, indicates that only one key may
	// be pressed at once.
	OneHot bool

	pressed map[string]bool
}

// ActionSpace returns a Bernoulli action space.
func (k *KeyActor) ActionSpace() ActionSpace {
	if k.OneHot {
		return anyrl.Softmax{}
	} else {
		return &anyrl.Bernoulli{}
	}
}

// ParamLen returns the number of keys.
func (k *KeyActor) ParamLen() int {
	return len(k.Keys)
}

// Reset resets pressed key information.
func (k *KeyActor) Reset() {
	k.pressed = map[string]bool{}
}

// Events generates key events.
func (k *KeyActor) Events(vec []float64) []interface{} {
	var events []interface{}

	for i, keyName := range k.Keys {
		press := vec[i] > 0.5
		if k.NoHold && press {
			evt := chrome.KeyEvents[keyName]
			evt1 := evt
			evt.Type = chrome.KeyDown
			evt1.Type = chrome.KeyUp
			events = append(events, &evt, &evt1)
		} else if !k.NoHold && press != k.pressed[keyName] {
			k.pressed[keyName] = press
			evt := chrome.KeyEvents[keyName]
			if press {
				evt.Type = chrome.KeyDown
			} else {
				evt.Type = chrome.KeyUp
			}
			events = append(events, &evt)
		}
	}

	return events
}

// Vectorize generates a vector for the key events.
func (k *KeyActor) Vectorize(events []interface{}) []float64 {
	if k.NoHold {
		k.pressed = map[string]bool{}
	}
	for _, event := range events {
		keyEvent, ok := event.(*chrome.KeyEvent)
		if !ok {
			continue
		}
		if keyEvent.Type == chrome.KeyDown {
			k.pressed[keyEvent.Code] = true
		} else if !k.NoHold {
			if keyEvent.Type == chrome.KeyUp {
				k.pressed[keyEvent.Code] = false
			}
		}
	}
	vector := make([]float64, len(k.Keys))
	for i, key := range k.Keys {
		if k.pressed[key] {
			vector[i] = 1
			if k.OneHot {
				break
			}
		}
	}
	return vector
}

// TapActor is an Actor which allows the agent to tap the
// middle of the screen.
type TapActor struct {
	// Screen dimensions.
	Width  int
	Height int

	// NoHold, if true, indicates that taps should be
	// instantaneous; the mouse cannot be held down.
	NoHold bool

	pressed bool
}

// ActionSpace returns a Bernoulli action space.
func (t *TapActor) ActionSpace() ActionSpace {
	return &anyrl.Bernoulli{}
}

// ParamLen returns 1.
func (t *TapActor) ParamLen() int {
	return 1
}

// Reset resets the pressed status.
func (t *TapActor) Reset() {
	t.pressed = false
}

// Events generates mouse events.
func (t *TapActor) Events(vec []float64) []interface{} {
	var events []interface{}

	press := vec[0] > 0.5

	if t.NoHold && press {
		evt := chrome.MouseEvent{
			Type:       chrome.MousePressed,
			X:          t.Width / 2,
			Y:          t.Height / 2,
			Button:     chrome.LeftButton,
			ClickCount: 1,
		}
		evt1 := evt
		evt1.Type = chrome.MouseReleased
		events = append(events, &evt, &evt1)
	} else if !t.NoHold && press != t.pressed {
		t.pressed = press
		evt := chrome.MouseEvent{
			Type:       chrome.MousePressed,
			X:          t.Width / 2,
			Y:          t.Height / 2,
			Button:     chrome.LeftButton,
			ClickCount: 1,
		}
		if !press {
			evt.Type = chrome.MouseReleased
		}
		events = append(events, &evt)
	}

	return events
}

// Vectorize generates a vector for the mouse events.
func (t *TapActor) Vectorize(events []interface{}) []float64 {
	if t.NoHold {
		t.pressed = false
	}
	for _, event := range events {
		mouseEvent, ok := event.(*chrome.MouseEvent)
		if !ok {
			continue
		}
		if mouseEvent.Type == chrome.MousePressed {
			t.pressed = true
		} else if !t.NoHold {
			if mouseEvent.Type == chrome.MouseReleased {
				t.pressed = false
			}
		}
	}
	vec := []float64{0}
	if t.pressed {
		vec[0] = 1
	}
	return vec
}

// MouseActor is an Actor which allows the agent to make
// arbitrary motions with the mouse.
//
// Mouse motions are performed relative to the current
// mouse position.
// In other words, the agent tells the environment to move
// the mouse by a given delta x and delta y.
type MouseActor struct {
	// Screen dimensions.
	Width  int
	Height int

	// NoHold, if true, indicates that clicks should be
	// instantaneous; the mouse cannot be held down.
	NoHold bool

	// Discrete, if true, limits mouse movements to a
	// pre-specified set of options.
	Discrete bool

	pressed bool
	lastX   int
	lastY   int
}

// ActionSpace returns an action space for producing
// arbitrary mouse actions.
func (m *MouseActor) ActionSpace() ActionSpace {
	if m.Discrete {
		size := len(m.options())
		return &anyrl.Tuple{
			Spaces:      []interface{}{&anyrl.Bernoulli{}, anyrl.Softmax{}},
			ParamSizes:  []int{1, size},
			SampleSizes: []int{1, size},
		}
	} else {
		return &anyrl.Tuple{
			Spaces:      []interface{}{&anyrl.Bernoulli{}, anyrl.Gaussian{}},
			ParamSizes:  []int{1, 4},
			SampleSizes: []int{1, 2},
		}
	}
}

// ParamLen returns the size of action parameter space.
func (m *MouseActor) ParamLen() int {
	if m.Discrete {
		return 1 + len(m.options())
	} else {
		return 5
	}
}

// Reset resets the mouse state.
func (m *MouseActor) Reset() {
	m.pressed = false
	m.lastX = m.Width / 2
	m.lastY = m.Height / 2
}

// Events generates mouse events.
func (m *MouseActor) Events(vec []float64) []interface{} {
	var events []interface{}

	press := vec[0] > 0.5
	x, y := m.mouseCoords(vec[1:])

	if m.NoHold && press {
		evt := chrome.MouseEvent{
			Type:       chrome.MousePressed,
			X:          m.lastX,
			Y:          m.lastY,
			Button:     chrome.LeftButton,
			ClickCount: 1,
		}
		evt1 := evt
		evt1.Type = chrome.MouseReleased
		events = append(events, &evt, &evt1)
	} else if !m.NoHold && press != m.pressed {
		m.pressed = press
		evt := chrome.MouseEvent{
			Type:       chrome.MousePressed,
			X:          m.lastX,
			Y:          m.lastY,
			Button:     chrome.LeftButton,
			ClickCount: 1,
		}
		if !press {
			evt.Type = chrome.MouseReleased
		}
		events = append(events, &evt)
	}

	if x != m.lastX || y != m.lastY {
		m.lastX = x
		m.lastY = y
		evt := &chrome.MouseEvent{
			Type:       chrome.MouseMoved,
			X:          x,
			Y:          y,
			ClickCount: 0,
		}
		if m.pressed {
			evt.Button = chrome.LeftButton
		}
		events = append(events, evt)
	}

	return events
}

// Vectorize generates a vector for the mouse events.
func (m *MouseActor) Vectorize(events []interface{}) []float64 {
	newX := m.lastX
	newY := m.lastY
	for _, event := range events {
		mouseEvent, ok := event.(*chrome.MouseEvent)
		if !ok {
			continue
		}
		newX = mouseEvent.X
		newY = mouseEvent.Y
		if mouseEvent.Type == chrome.MousePressed {
			m.pressed = true
			if m.NoHold {
				// Go with the first click we find.
				break
			}
		} else if !m.NoHold {
			if mouseEvent.Type == chrome.MouseReleased {
				m.pressed = false
			}
		}
	}
	var vec []float64
	if m.Discrete {
		vec = make([]float64, 1+len(m.options()))
		vec[1+m.closestOption(newX-m.lastX, newY-m.lastY)] = 1
	} else {
		vec = []float64{0, 4 * float64(newX-m.lastX) / float64(m.Width),
			4 * float64(newY-m.lastY) / float64(m.Height)}
	}
	m.lastX = newX
	m.lastY = newY
	if m.pressed {
		vec[0] = 1
	}
	return vec
}

func (m *MouseActor) mouseCoords(vec []float64) (x, y int) {
	if m.Discrete {
		for i, val := range vec {
			if val != 0 {
				offset := m.options()[i]
				x = m.lastX + offset[0]
				y = m.lastY + offset[1]
				break
			}
		}
	} else {
		fx, fy := vec[0], vec[1]
		x = m.lastX + int(essentials.Round(float64(m.Width)*fx/4))
		y = m.lastY + int(essentials.Round(float64(m.Height)*fy/4))
	}
	x = essentials.MaxInt(0, essentials.MinInt(m.Width-1, x))
	y = essentials.MaxInt(0, essentials.MinInt(m.Height-1, y))
	return
}

// options returns the (x, y) offsets for each of the
// discrete options.
func (m *MouseActor) options() [][2]int {
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

// closestOption finds the discrete option which is the
// closest to approximating the given delta.
func (m *MouseActor) closestOption(deltaX, deltaY int) int {
	opts := m.options()
	var closestOpt int
	closestDist := math.Inf(1)
	for i, opt := range opts {
		dist := math.Sqrt(math.Pow(float64(deltaX-opt[0]), 2) +
			math.Pow(float64(deltaY-opt[1]), 2))
		if dist < closestDist {
			closestDist = dist
			closestOpt = i
		}
	}
	return closestOpt
}
