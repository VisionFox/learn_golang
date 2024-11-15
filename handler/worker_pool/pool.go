package worker_pool

import (
	"sync"
	"time"
)

type Pool struct {
	JobQueue   chan Job
	dispatcher *dispatcher
	wg         sync.WaitGroup
}

// 打包任务
func (p *Pool) wrapJob(job func()) func() {
	return func() {
		defer p.JobDone()
		job()
	}
}

func (p *Pool) SendJobWithTimeout(job func(), t time.Duration) bool {
	select {
	case <-time.After(t):
		return false
	case p.JobQueue <- p.wrapJob(job):
		p.WaitCount(1)
		return true
	}
}

func (p *Pool) SendJobWithDeadline(job func(), t time.Time) bool {
	s := t.Sub(time.Now())
	if s <= 0 {
		s = time.Second // timeout
	}
	select {
	case <-time.After(s):
		return false
	case p.JobQueue <- p.wrapJob(job):
		p.WaitCount(1)
		return true
	}
}

// SendJob 发送任务
func (p *Pool) SendJob(job func()) {
	p.WaitCount(1)
	p.JobQueue <- p.wrapJob(job)
}

func (p *Pool) JobDone() {
	p.wg.Done()
}

func (p *Pool) WaitCount(count int) {
	p.wg.Add(count)
}

// WaitAll 等待所有goroutine退出
func (p *Pool) WaitAll() {
	p.wg.Wait()
}

// Release 释放资源
func (p *Pool) Release() {
	p.dispatcher.stop <- struct{}{}
	<-p.dispatcher.stop
}
