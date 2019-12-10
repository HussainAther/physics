# Python processes

Various physical processes for stochastic differential equations

## Shared parameters

* `rate` : float > 0 that define the inter-arrival rate. 
* `startTime` : (default 0) The start time of the process
* `startPosition` : (default 0) The starting position of the process at startTime
* `conditional processes` :
  * If the process is known at some future time, endPosition and endTime 
    condition the process to arrive there.
* `endTime` : (default None) the time in the future the process is known at, > startTime
* `endPosition` : (default None) the position the process is at in the future.
                  > startPosition and equal to startPosition + int. 
