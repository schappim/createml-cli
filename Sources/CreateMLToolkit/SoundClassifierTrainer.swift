import Foundation
import CreateML
import CoreML
import SoundAnalysis

/// Trains sound classification models
public class SoundClassifierTrainer {

    public struct TrainingParameters {
        public var overlapFactor: Double
        public var validationData: URL?

        public init(
            overlapFactor: Double = 0.5,
            validationData: URL? = nil
        ) {
            self.overlapFactor = overlapFactor
            self.validationData = validationData
        }
    }

    public struct TrainingResult {
        public let modelURL: URL
        public let trainingAccuracy: Double
        public let validationAccuracy: Double?
        public let trainingDuration: TimeInterval
        public let classLabels: [String]
    }

    public init() {}

    /// Train a sound classifier from a directory of labeled audio files
    /// - Parameters:
    ///   - trainingDataURL: Directory containing subdirectories for each class with audio files
    ///   - outputURL: Where to save the trained .mlmodel
    ///   - parameters: Training parameters
    ///   - progressHandler: Called with progress updates
    /// - Returns: Training result with metrics
    public func train(
        trainingDataURL: URL,
        outputURL: URL,
        author: String? = nil,
        description: String? = nil,
        parameters: TrainingParameters = TrainingParameters(),
        progressHandler: ((String) -> Void)? = nil
    ) throws -> TrainingResult {
        let startTime = Date()

        progressHandler?("Loading training data from \(trainingDataURL.path)...")

        let trainingData = MLSoundClassifier.DataSource.labeledDirectories(at: trainingDataURL)

        var modelParameters = MLSoundClassifier.ModelParameters(overlapFactor: parameters.overlapFactor)

        if let validationURL = parameters.validationData {
            let validationData = MLSoundClassifier.DataSource.labeledDirectories(at: validationURL)
            modelParameters.validation = .dataSource(validationData)
        }

        progressHandler?("Training sound classifier...")

        let classifier = try MLSoundClassifier(trainingData: trainingData, parameters: modelParameters)

        let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
        var validationAccuracy: Double? = nil
        let valError = classifier.validationMetrics.classificationError
        if !valError.isNaN {
            validationAccuracy = (1.0 - valError) * 100
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Sound classifier trained with CreateML CLI",
            version: "1.0"
        )

        try classifier.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        let classLabels = getClassLabels(from: trainingDataURL)

        return TrainingResult(
            modelURL: outputURL,
            trainingAccuracy: trainingAccuracy,
            validationAccuracy: validationAccuracy,
            trainingDuration: trainingDuration,
            classLabels: classLabels
        )
    }

    private func getClassLabels(from url: URL) -> [String] {
        let fileManager = FileManager.default
        guard let contents = try? fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey]) else {
            return []
        }
        return contents
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .map { $0.lastPathComponent }
            .sorted()
    }
}
