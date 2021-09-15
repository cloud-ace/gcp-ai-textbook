package main

import (
	"context"
	"fmt"
	"log"

	"cloud.google.com/go/translate"
	"golang.org/x/text/language"
)

func main() {
	ctx := context.Background()

	client, err := translate.NewClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	// 1
	ls, err := client.SupportedLanguages(ctx, language.Japanese)
	if err != nil {
		log.Fatalf("Failed to get of supported languages: %v", err)
	}

	// 2
	fmt.Printf("Translation APIは%dの言語に対応しています\n", len(ls))
	for _, l := range ls {
		fmt.Printf("言語: %s, タグ: %v\n", l.Name, l.Tag)
	}
}
