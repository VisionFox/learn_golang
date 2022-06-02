package main

import "errors"

var OPERATION1FAILED error = errors.New("1")
var OPERATION2FAILED error = errors.New("2")
var OPERATION3FAILED error = errors.New("3")
var OPERATION4FAILED error = errors.New("4")

func Handle() error {
	var err error
	if Operation1() {
		if Operation2() {
			if Operation3() {
				if Operation4() {
					// do
				} else {
					err = OPERATION4FAILED
				}
			} else {
				err = OPERATION3FAILED
			}
		} else {
			err = OPERATION2FAILED
		}
	} else {
		err = OPERATION1FAILED
	}
	return err
}

func HandlePlus() error {
	var err error

	for true {
		if !Operation1() {
			err = OPERATION1FAILED
			break
		}

		if !Operation2() {
			err = OPERATION1FAILED
			break
		}

		if !Operation3() {
			err = OPERATION1FAILED
			break
		}

		if !Operation4() {
			err = OPERATION1FAILED
			break
		}

		// do

		break
	}

	return err
}

func Operation1() bool {
	return false
}

func Operation2() bool {
	return false
}

func Operation3() bool {
	return false
}

func Operation4() bool {
	return false
}
