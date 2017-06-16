package main

import (
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
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
	Events(actVec anyvec.Vector) []interface{}

	// Vectorize performs the pseudo-inverse of Events.
	// If no action vector could produce the events, a
	// reasonable vector should be chosen which would
	// produce some of the events.
	Vectorize(c anyvec.Creator, events []interface{}) anyvec.Vector
}

// KeyActor is an Actor which produces keyboard events
// where each key is controlled by another vector
// component.
type KeyActor struct {
	Keys []string

	// NoHold, if true, indicates that key presses are
	// instantaneous; keys cannot be held down.
	NoHold bool

	pressed map[string]bool
}

// ActionSpace returns a Bernoulli action space.
func (k *KeyActor) ActionSpace() ActionSpace {
	return &anyrl.Bernoulli{}
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
func (k *KeyActor) Events(vec anyvec.Vector) []interface{} {
	var events []interface{}

	ops := vec.Creator().NumOps()
	thresh := vec.Creator().MakeNumeric(0.5)

	for i, keyName := range k.Keys {
		press := ops.Greater(anyvec.Sum(vec.Slice(i, i+1)), thresh)
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
func (k *KeyActor) Vectorize(c anyvec.Creator, events []interface{}) anyvec.Vector {
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
		}
	}
	return c.MakeVectorData(c.MakeNumericList(vector))
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
func (t *TapActor) Events(vec anyvec.Vector) []interface{} {
	var events []interface{}

	ops := vec.Creator().NumOps()
	thresh := vec.Creator().MakeNumeric(0.5)
	press := ops.Greater(anyvec.Sum(vec.Slice(0, 1)), thresh)

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
func (t *TapActor) Vectorize(c anyvec.Creator, events []interface{}) anyvec.Vector {
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
	return c.MakeVectorData(c.MakeNumericList(vec))
}
