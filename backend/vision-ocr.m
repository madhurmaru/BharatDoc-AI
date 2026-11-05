#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) { fprintf(stderr, "Usage: vision-ocr image\n"); return 2; }
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = YES;
        NSError *error = nil;
        NSArray<NSString *> *supported = [request supportedRecognitionLanguagesAndReturnError:&error];
        if (strcmp(argv[1], "--languages") == 0) {
            for (NSString *language in supported) puts(language.UTF8String);
            return 0;
        }
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        NSArray<NSString *> *preferred = @[@"hi-IN", @"en-US", @"bn-IN", @"ta-IN", @"te-IN", @"mr-IN"];
        NSMutableArray<NSString *> *chosen = [NSMutableArray array];
        for (NSString *language in preferred) if ([supported containsObject:language]) [chosen addObject:language];
        if (chosen.count > 0) request.recognitionLanguages = chosen;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithURL:url options:@{}];
        if (![handler performRequests:@[request] error:&error]) {
            fprintf(stderr, "OCR error: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        for (VNRecognizedTextObservation *observation in request.results) {
            VNRecognizedText *text = [[observation topCandidates:1] firstObject];
            if (text) puts(text.string.UTF8String);
        }
    }
    return 0;
}
