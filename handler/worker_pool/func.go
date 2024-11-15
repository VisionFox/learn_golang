package worker_pool

import (
	"fmt"
	"log"
	"time"
)

type Job func()

// 真正执行任务
func runJob(f func()) {
	defer func() {
		if err := recover(); err != nil {
			log.Printf("gpool Job panic err: %v", err)
		}
	}()

	f()
}

// 初始化worker
func newWorker(pool chan *worker) *worker {
	return &worker{
		workerPool: pool,
		jobChannel: make(chan Job),
		stop:       make(chan struct{}),
	}
}

// 初始化分配器
func newDispatcher(workerPool chan *worker, jobQueue chan Job) *dispatcher {
	d := &dispatcher{
		workerPool: workerPool,
		jobQueue:   jobQueue,
		stop:       make(chan struct{}),
	}

	for i := 0; i < cap(d.workerPool); i++ {
		worker := newWorker(d.workerPool)
		worker.start()
	}

	go d.dispatch()
	return d
}

// NewPool 初始化goroutine Pool
func NewPool(numWorkers int, jobQueueLen int) *Pool {
	jobQueue := make(chan Job, jobQueueLen)
	workerPool := make(chan *worker, numWorkers)

	pool := &Pool{
		JobQueue:   jobQueue,
		dispatcher: newDispatcher(workerPool, jobQueue),
	}

	return pool
}

func main() {
	// 初始化 10个worker(goroutine) 任务队列长度是1000
	var pool = NewPool(10, 1000)

	pool.SendJobWithTimeout(func() {
		fmt.Println("SendJobWithTimeout")
	}, 2*time.Second)

	// 发送任务
	pool.SendJob(func() {
		fmt.Println("send job")
	})

	pool.SendJobWithDeadline(func() {
		fmt.Println("SendJobWithDeadline")
	}, time.Now().Add(time.Second*3))

	// 等待资源释放和退出
	pool.WaitAll()
	pool.Release()
}
