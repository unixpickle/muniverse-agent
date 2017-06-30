# µniverse-agent

This program trains RL agents to play games in [µniverse](https://github.com/unixpickle/muniverse). It uses [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) to optimize neural networks policies.

This is intended to be used as a baseline. It gives an idea of how well traditional deep-RL does on µniverse games.

# Using CUDA

This package uses [anyplugin](https://github.com/unixpickle/anyplugin). Thus, you can add `-tags cuda` to any `go run`, `go build`, or `go get` command to enable CUDA. You will want to see the [cuda build instructions](https://godoc.org/github.com/unixpickle/cuda#hdr-Building). You will also want to download muniverse-agent with its CUDA dependencies:

```
go get -u -tags cuda github.com/unixpickle/muniverse-agent
```
