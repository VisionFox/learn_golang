package main

import "learn_golang/handler"

func main() {
	handler.SayHello()

	a := []int{1, 2, 4}
	println(candy(a))
	b := [][]int{{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}}
	println(reconstructQueue(b))
}
