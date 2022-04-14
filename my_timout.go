package main

import (
	"fmt"
	"strconv"
	"sync"
	"time"
)

func simpleTimeout() {
	msg := make(chan int, 10)
	defer close(msg)
	done := make(chan bool, 0)
	defer close(done)
	wg := new(sync.WaitGroup)

	// receive
	wg.Add(1)
	go func() {
		defer func() {
			msg <- -1
			wg.Done()
		}()

		ticker := time.After(3 * time.Second)

		for {
			select {
			case <-ticker:
				fmt.Println("ticker timeout")
				return
			case <-done:
				fmt.Println("done true")
				return
			default:
				fmt.Println("msg : " + strconv.Itoa(<-msg))
			}
		}
	}()

	// send
	wg.Add(1)
	for i := 0; i < 10; i++ {
		msg <- i
		time.Sleep(1 * time.Second)
	}
	done <- true
	wg.Done()
	wg.Wait()
}
