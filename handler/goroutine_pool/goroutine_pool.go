package goroutine_pool

import (
	"errors"
	"log"
	"sync"
)

type GoroutinePool struct {
	pool    chan struct{}
	maxSize int
	closed  bool
	mu      sync.Mutex
	wg      *sync.WaitGroup
}

func NewGoroutinePool(maxSize int) *GoroutinePool {
	return &GoroutinePool{
		pool:    make(chan struct{}, maxSize),
		maxSize: maxSize,
	}
}

func (gp *GoroutinePool) Resize(size int) {
	gp.maxSize = size
	gp.pool = make(chan struct{}, size)
}

func (gp *GoroutinePool) Submit(task func()) error {
	gp.mu.Lock()
	if gp.closed {
		gp.mu.Unlock()
		return errors.New("goroutine pool is closed")
	}
	gp.wg.Add(1)
	gp.mu.Unlock()

	//提交 信号入队 满了就阻塞
	gp.pool <- struct{}{}
	go func() {
		defer func() {
			// 运行完毕 信号出队
			<-gp.pool
			gp.wg.Done()
		}()
		task()
	}()
	return nil
}

func (gp *GoroutinePool) Shutdown() {
	gp.mu.Lock()
	gp.closed = true
	gp.mu.Unlock()

	gp.wg.Wait()
	close(gp.pool)
}

// SafeSubmit 保证提交任务的安全性 抛出error 由调用者处理 me
func (gp *GoroutinePool) SafeSubmit(task func()) error {
	return gp.Submit(func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Recovered in task: %v", r)
			}
		}()
		task()
	})
}

// SafeSubmit 保证提交任务的安全性 抛出error 由调用者处理 other
func SafeSubmit(pool *GoroutinePool, task func()) error {
	return pool.Submit(func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Recovered in task: %v", r)
			}
		}()
		task()
	})
}
