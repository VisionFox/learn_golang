package main

//func main() {
//	//wg := new(sync.WaitGroup)
//	//runtime.GOMAXPROCS(1)
//	//wg.Add(3)
//	//for i := 0; i < 3; i++ {
//	//	go func(num int) {
//	//		fmt.Println(num)
//	//		wg.Done()
//	//	}(i)
//	//}
//	//wg.Wait()
//	//
//	//time.Sleep(100 * time.Second)
//	//
//	//fmt.Printf("%08b\n", ipToInt("1.2.3.4"))
//	//fmt.Printf("%d\n", ipToInt("1.2.3.4"))
//
//	s := "com.futu.hk"
//	ss := make([]byte, 0)
//	for i := 0; i < len(s); i++ {
//		ss = append(ss, s[i])
//	}
//	print(test(ss))
//}
//
//func test(source []byte) string {
//	if len(source) == 0 {
//		return ""
//	}
//
//	source = reserve(source, 0, len(source)-1)
//
//	start := 0
//	for i := 0; i < len(source); i++ {
//		if source[i] == byte('.') {
//			source = reserve(source, start, i-1)
//			start = i + 1
//		}
//	}
//
//	source = reserve(source, start, len(source)-1)
//
//	return string(source)
//}
//
//func reserve(source []byte, start, end int) []byte {
//	for start < end {
//		source[start], source[end] = source[end], source[start]
//		start++
//		end--
//	}
//
//	return source
//}

func main() {
	s := "com.futu5.hk.fund"
	ss := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		ss = append(ss, s[i])
	}
	println(test(ss))
}

func test(source []byte) string {
	if len(source) == 0 {
		return ""
	}

	source = reserve(source, 0, len(source)-1)

	start := 0
	for i := 0; i < len(source); i++ {
		if source[i] == byte('.') {
			source = reserve(source, start, i-1)
			start = i + 1
		}
	}

	source = reserve(source, start, len(source)-1)

	return string(source)
}

func reserve(source []byte, start, end int) []byte {
	for start < end {
		source[start], source[end] = source[end], source[start]
		start++
		end--
	}

	return source
}
