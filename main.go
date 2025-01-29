package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -L/usr/local/lib -linference -ljson-c -lsentencepiece -lonnxruntime -lonnxruntime_providers_shared -lonnxruntime_providers_cuda -lcuda -lcudart -lm -lstdc++ -licuuc -licudata
#include "inference.h"
*/
import "C"
import (
	"fmt"
	"time"
	"unsafe"
)

func main() {
	geco := C.NewGeco(0, 0)
	defer C.FreeGeco(geco)
	fmt.Printf("Geco Type: %T\n", geco) // Type: unsafe.Pointer

	/* for i, test := range tests {
		speedTestGo(geco, i+1, test)
		if i == 0 {
			break
		}
	} */
	//speedTestGo(geco, 1, tests[0])
	//x := []string{"Hello, world.", "\\x1b[32mAdditional Sentence.", "\\x1b[0mMy name is John."}
	//res := runInference(geco, x)
	//fmt.Printf("Result: %q\n", res)
	testNewGibb(geco)
	//testGibb(geco, 1, tests[0])
}

func testNewGibb(geco unsafe.Pointer) {
	// Allocate memory for the 2D array in Go
	probs := make([][C.GIBB_CLASSES]float64, C.MAX_BATCH_SIZE)

	testTexts := []string{"Hello, world.", "blue red yellow green orange", "John was here."}
	num_texts := len(testTexts)
	// Convert Go strings to C strings
	cTexts := make([]*C.char, num_texts)
	for i, text := range testTexts {
		cTexts[i] = C.CString(text)             // Convert Go string to C string
		defer C.free(unsafe.Pointer(cTexts[i])) // Ensure C strings are freed after use
	}

	// Convert Go 2D array to C 2D array
	cProbs := (*[C.GIBB_CLASSES]C.double)(unsafe.Pointer(&probs[0])) // Type: *[4]._Ctype_double
	fmt.Printf("cProbs Type: %T\n", cProbs)
	//_ = cProbs

	C.GecoGibb(geco, cProbs, &cTexts[0], C.int(num_texts)) // Type: *main._Ctype_char

	fmt.Println("Probs array after calling C function:")
	for i := 0; i < num_texts; i++ {
		fmt.Println(probs[i])
		fmt.Printf("  > Type of Probs[%v][0] = %T\n", i, probs[i][0])
	}
}

func testGibb(geco unsafe.Pointer, testNum int, test TestCase) {
	//gibbClasses := []string{"Clean", "Mild Gibberish", "Word Salad", "Noise"}

	//res := runGibb(geco, test.Texts)
	res := []GibbResults{}
	for i, v := range res {
		fmt.Printf("\nScores[%v]:\n", i)
		fmt.Printf("   - \"Clean\": %.5f\n", v.Score.Clean)
		fmt.Printf("   - \"Mild\": %.5f\n", v.Score.Mild)
		fmt.Printf("   - \"Word Salad\": %.5f\n", v.Score.WordSalad)
		fmt.Printf("   - \"Noise\": %.5f\n", v.Score.Noise)
		/* for j := 0; j < 4; j++ {
			fmt.Printf("\t\"Clean\": %.5f\n", gibbClasses[j], v.Score.Clean[j])
		} */
	}
}
func speedTestGo(geco unsafe.Pointer, testNum int, test TestCase) {
	margin := 0.02

	start := time.Now()
	res := runInference(geco, test.Texts)
	//res := GecoRun(geco, test.Texts, test.BatchSize)
	totalTime := time.Now().Sub(start).Seconds() // Calculate the elapsed time

	timeDiff := test.AvgTimeGpu - totalTime
	if timeDiff > margin {
		fmt.Printf("Test #%v - Time taken: \x1b[1;32m%.3f\x1b[0m\n", testNum, totalTime)
	} else if timeDiff < -margin {
		fmt.Printf("Test #%v - Time taken: \x1b[1;31m%.3f\x1b[0m\n", testNum, totalTime)
	} else {
		fmt.Printf("Test #%v - Time taken: %.3f\n", testNum, totalTime)
	}

	if res == test.ExpectText {
		fmt.Printf("Test #%v - Result: \x1b[1;32mPASS\x1b[0m\n", testNum)
	} else {
		fmt.Printf("Test #%v - Result: \x1b[1;31mFAIL\x1b[0m\n\tResult: %q", testNum, res)
	}
	/* if !reflect.DeepEqual(res, test.Expect) {
		fmt.Printf("Expected: %v\n", test.Expect)
		fmt.Printf("Actual: %v\n", res)
	} */
	fmt.Printf("\n")
}

func runInference(geco unsafe.Pointer, testTexts []string) string {
	testSize := len(testTexts)

	// Convert Go strings to C strings
	cTexts := make([]*C.char, testSize)
	for i, text := range testTexts {
		cTexts[i] = C.CString(text)             // Convert Go string to C string
		defer C.free(unsafe.Pointer(cTexts[i])) // Ensure C strings are freed after use
	}

	result := C.GecoRun(geco, &cTexts[0], C.int(testSize)) // Type: *main._Ctype_char
	fmt.Printf("Type of result: %T\n\n", result)
	if result == nil {
		fmt.Println("Error: GecoRun returned a null pointer")
		return ""
	}

	// Convert the C char* result to a Go string
	goResult := C.GoString(result)
	defer C.free(unsafe.Pointer(result))

	return goResult
}

// TODO: Run Gibberish
/* func runGibb(geco unsafe.Pointer, testTexts []string) []GibbResults {
	num_texts := len(testTexts)

	// Convert Go strings to C strings
	cTexts := make([]*C.char, num_texts)
	for i, text := range testTexts {
		cTexts[i] = C.CString(text)             // Convert Go string to C string
		defer C.free(unsafe.Pointer(cTexts[i])) // Ensure C strings are freed after use
	}

	gibb_result := C.GecoGibb(geco, &cTexts[0], C.int(num_texts)) // Type: *main._Ctype_char
	fmt.Printf("Type of result: %T\n\n", gibb_result)
	if gibb_result == nil {
		fmt.Println("Error: GecoGibb returned a null pointer")
		return nil
	}

	var gibb_scores []GibbResults
	for i := 0; i < num_texts; i++ {
		var gibb GibbResults
		// Get the pointer to the ith row
		rowPtr := *(**C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(gibb_result)) + uintptr(i)*unsafe.Sizeof(gibb_result)))

		// Dummy values
		gibb.Index = 0
		gibb.Length = 0

		// Create a Go slice for the ith row
		gibb.Score.Clean = float32(*(*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(rowPtr)) + uintptr(0)*unsafe.Sizeof(*rowPtr))))
		gibb.Score.Mild = float32(*(*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(rowPtr)) + uintptr(1)*unsafe.Sizeof(*rowPtr))))
		gibb.Score.WordSalad = float32(*(*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(rowPtr)) + uintptr(2)*unsafe.Sizeof(*rowPtr))))
		gibb.Score.Noise = float32(*(*C.double)(unsafe.Pointer(uintptr(unsafe.Pointer(rowPtr)) + uintptr(3)*unsafe.Sizeof(*rowPtr))))
		gibb_scores = append(gibb_scores, gibb)
	}

	return gibb_scores
} */

func tmp() {
	var x GibbResults
	x.Index = 0
	x.Length = 0
	x.Score.Clean = 92.8

}
