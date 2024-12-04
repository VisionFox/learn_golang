package handler

import (
	"sync"
)

func MyPrintCatDogFish() {
	chA := make(chan int, 0)
	chB := make(chan int, 0)
	chC := make(chan int, 0)

	wg := sync.WaitGroup{}
	wg.Add(3)

	go cat(&wg, 100, chA, chB)
	go dog(&wg, 100, chB, chC)
	go fish(&wg, 100, chC, chA)
	chA <- 0

	wg.Wait()
}

func cat(wg *sync.WaitGroup, targetCnt int, chIn chan int, chOut chan int) {
	defer wg.Done()
	for {
		select {
		case cnt, ok := <-chIn:
			if !ok {
				close(chOut)
				return
			}
			if cnt >= targetCnt {
				close(chOut)
				return
			}
			println("cat")
			chOut <- cnt + 1
		}
	}
}

func dog(wg *sync.WaitGroup, targetCnt int, chIn chan int, chOut chan int) {
	defer wg.Done()
	for {
		select {
		case cnt, ok := <-chIn:
			if !ok {
				close(chOut)
				return
			}
			if cnt >= targetCnt {
				close(chOut)
				return
			}
			println("dog")
			chOut <- cnt + 1
		}
	}
}

func fish(wg *sync.WaitGroup, targetCnt int, chIn chan int, chOut chan int) {
	defer wg.Done()
	for {
		select {
		case cnt, ok := <-chIn:
			if !ok {
				close(chOut)
				return
			}
			if cnt >= targetCnt {
				close(chOut)
				return
			}
			println("fish")
			chOut <- cnt + 1
		}
	}
}
