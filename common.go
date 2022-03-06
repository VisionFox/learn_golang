package main

import "errors"

type ListNode struct {
	Val  int
	Next *ListNode
}

type IList interface {
}

type IStack interface {
	Len() int
	Push(v interface{})
	Pop() interface{}
	IsEmpty() bool
}

type Stack struct {
	items []interface{}
}

func NewStack() *Stack {
	stack := &Stack{
		items: make([]interface{}, 0),
	}
	return stack
}

func (s *Stack) Len() int {
	return len(s.items)
}

func (s *Stack) Push(v interface{}) {
	s.items = append(s.items, v)
}

func (s *Stack) Pop() (interface{}, error) {
	if s.IsEmpty() {
		return nil, errors.New("stack empty")
	}
	top := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return top, nil
}

func (s *Stack) IsEmpty() bool {
	if len(s.items) == 0 {
		return true
	}
	return false
}

func majorityElement(nums []int) int {
	numFlag, cnt := nums[0], 1

	for _, num := range nums {
		if num == numFlag {
			cnt++
		} else {
			cnt--
			if cnt == 0 {
				numFlag = num
				cnt++
			}
		}
	}

	return numFlag
}

func calculate(s string) (ans int) {
	stack := make([]int, 0)
	preSign := '+'
	curNum := 0

	for idx, char := range s {
		isDigit := char >= '0' && char <= '9'
		if isDigit {
			curNum = curNum*10 + int(char-'0')
		}

		if !isDigit && char != ' ' || idx == len(s)-1 {
			switch preSign {
			case '+':
				stack = append(stack, curNum)
			case '-':
				stack = append(stack, -curNum)
			case '*':
				stack[len(stack)-1] *= curNum
			default:
				stack[len(stack)-1] /= curNum
			}
			preSign = char
			curNum = 0
		}
	}

	for _, v := range stack {
		ans += v
	}
	return ans
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

func min(i, j int) int {
	if i > j {
		return j
	}
	return i
}
