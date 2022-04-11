package main

import (
	"fmt"
	"strconv"
	"sync"
)

func goroutine1(ch chan int, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := 1; i <= 100; i++ {
		if i%2 == 1 {
			fmt.Println("g_1 : " + strconv.Itoa(i))
		} else {
			ch <- i
		}
	}
	close(ch)
}

func goroutine2(notify chan int, wg *sync.WaitGroup) {
	defer wg.Done()

	for num := range notify {
		fmt.Println("g_2 : " + strconv.Itoa(num))
	}
}

func printNum() {
	msg := make(chan int, 0)
	wg := new(sync.WaitGroup)
	wg.Add(2)
	go goroutine1(msg, wg)
	go goroutine2(msg, wg)
	wg.Wait()
}

func work(receiveCh, sendCh chan int, symbol int, wg *sync.WaitGroup, n int) {
	defer wg.Done()

	for num := range receiveCh {
		if num > n {
			break
		}

		fmt.Println("goroutine:", symbol, "print", num)
		sendCh <- num + 1
	}

	close(sendCh)
	fmt.Println("goroutine: finish", symbol)
}

func printNumPlus(nGoroutine, n int) {
	wg := new(sync.WaitGroup)
	once := new(sync.Once)

	sendOrReceiveChs := make([]chan int, nGoroutine)
	for i := 0; i < nGoroutine; i++ {
		sendOrReceiveChs[i] = make(chan int)
	}

	for i := 0; i < nGoroutine; i++ {
		receiveCh := sendOrReceiveChs[i]
		// 记住是 +1
		sendCh := sendOrReceiveChs[(i+nGoroutine+1)%nGoroutine]
		wg.Add(1)
		go work(receiveCh, sendCh, i+1, wg, n)

		once.Do(func() {
			receiveCh <- 1
		})
	}
	wg.Wait()
}
