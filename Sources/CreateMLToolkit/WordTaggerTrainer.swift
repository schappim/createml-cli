import Foundation
import CreateML
import CoreML

/// Trains word tagging models for NER, POS tagging, etc.
public class WordTaggerTrainer {

    public struct TrainingParameters {
        public var validationData: URL?
        public var language: String?

        public init(
            validationData: URL? = nil,
            language: String? = nil
        ) {
            self.validationData = validationData
            self.language = language
        }
    }

    public struct TrainingResult {
        public let modelURL: URL
        public let trainingAccuracy: Double
        public let validationAccuracy: Double?
        public let trainingDuration: TimeInterval
        public let tagLabels: [String]
    }

    public init() {}

    /// Train a word tagger from a JSON file with token annotations
    /// - Parameters:
    ///   - trainingDataURL: JSON file with tokens and labels
    ///   - outputURL: Where to save the trained .mlmodel
    ///   - tokenColumn: Name of the tokens column
    ///   - labelColumn: Name of the labels column
    ///   - parameters: Training parameters
    ///   - progressHandler: Called with progress updates
    /// - Returns: Training result with metrics
    public func train(
        trainingDataURL: URL,
        outputURL: URL,
        tokenColumn: String = "tokens",
        labelColumn: String = "labels",
        author: String? = nil,
        description: String? = nil,
        parameters: TrainingParameters = TrainingParameters(),
        progressHandler: ((String) -> Void)? = nil
    ) throws -> TrainingResult {
        let startTime = Date()

        progressHandler?("Loading training data from \(trainingDataURL.path)...")

        let trainingData = try MLDataTable(contentsOf: trainingDataURL)

        progressHandler?("Found \(trainingData.rows.count) training examples...")

        progressHandler?("Training word tagger...")

        let tagger: MLWordTagger

        if let validationURL = parameters.validationData {
            let validationData = try MLDataTable(contentsOf: validationURL)
            let modelParams = MLWordTagger.ModelParameters(validationData: validationData)
            tagger = try MLWordTagger(
                trainingData: trainingData,
                tokenColumn: tokenColumn,
                labelColumn: labelColumn,
                parameters: modelParams
            )
        } else {
            tagger = try MLWordTagger(
                trainingData: trainingData,
                tokenColumn: tokenColumn,
                labelColumn: labelColumn
            )
        }

        let trainingAccuracy = (1.0 - tagger.trainingMetrics.taggingError) * 100
        var validationAccuracy: Double? = nil
        let valError = tagger.validationMetrics.taggingError
        if !valError.isNaN {
            validationAccuracy = (1.0 - valError) * 100
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Word tagger trained with CreateML CLI",
            version: "1.0"
        )

        try tagger.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        // Get unique tag labels from the data
        let tagLabels = try getTagLabels(from: trainingDataURL, labelColumn: labelColumn)

        return TrainingResult(
            modelURL: outputURL,
            trainingAccuracy: trainingAccuracy,
            validationAccuracy: validationAccuracy,
            trainingDuration: trainingDuration,
            tagLabels: tagLabels
        )
    }

    private func getTagLabels(from url: URL, labelColumn: String) throws -> [String] {
        let data = try Data(contentsOf: url)
        guard let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }

        var labels = Set<String>()
        for item in jsonArray {
            if let tagList = item[labelColumn] as? [String] {
                for tag in tagList {
                    labels.insert(tag)
                }
            }
        }
        return Array(labels).sorted()
    }
}
