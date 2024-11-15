package worker_pool

// 创建真正执行任务的worker
type worker struct {
	workerPool chan *worker
	jobChannel chan Job
	stop       chan struct{}
}

// 开始执行任务
func (w *worker) start() {
	go func() {
		var job Job
		for {
			// worker free, add it to pool
			w.workerPool <- w

			select {
			case job = <-w.jobChannel:
				runJob(job)
			case <-w.stop:
				w.stop <- struct{}{}
				return
			}
		}
	}()
}
