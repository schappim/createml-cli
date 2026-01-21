import Foundation
import CreateML
import CoreML

/// Trains object detection models
public class ObjectDetectorTrainer {

    public struct TrainingParameters {
        public var maxIterations: Int
        public var validationData: URL?
        public var batchSize: Int

        public init(
            maxIterations: Int = 500,
            validationData: URL? = nil,
            batchSize: Int = 8
        ) {
            self.maxIterations = maxIterations
            self.validationData = validationData
            self.batchSize = batchSize
        }
    }

    public struct TrainingResult {
        public let modelURL: URL
        public let trainingMAP: Double  // Mean Average Precision @ IoU 0.5
        public let validationMAP: Double?
        public let trainingDuration: TimeInterval
        public let classLabels: [String]
    }

    public init() {}

    /// Train an object detector from annotated images
    /// - Parameters:
    ///   - trainingDataURL: Directory containing annotations.json and images
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

        // Check for annotations file
        let annotationsURL = trainingDataURL.appendingPathComponent("annotations.json")
        guard FileManager.default.fileExists(atPath: annotationsURL.path) else {
            throw TrainingError.missingAnnotations(
                "annotations.json not found in \(trainingDataURL.path). " +
                "Create an annotations.json file with bounding box annotations."
            )
        }

        // Load data using the directory annotation type
        let trainingData = MLObjectDetector.DataSource.directoryWithImagesAndJsonAnnotation(at: trainingDataURL)

        progressHandler?("Configuring training parameters...")

        var modelParameters = MLObjectDetector.ModelParameters(
            batchSize: parameters.batchSize,
            maxIterations: parameters.maxIterations
        )

        if let validationURL = parameters.validationData {
            let validationData = MLObjectDetector.DataSource.directoryWithImagesAndJsonAnnotation(at: validationURL)
            modelParameters = MLObjectDetector.ModelParameters(
                validation: .dataSource(validationData),
                batchSize: parameters.batchSize,
                maxIterations: parameters.maxIterations
            )
        }

        progressHandler?("Training object detector (max \(parameters.maxIterations) iterations)...")
        progressHandler?("This may take a while depending on your dataset size...")

        let detector = try MLObjectDetector(
            trainingData: trainingData,
            parameters: modelParameters,
            annotationType: .boundingBox()
        )

        // Get training metrics - use mean average precision (mAP) for object detection
        let trainingMetrics = detector.trainingMetrics.meanAveragePrecision
        let trainingMAPValue = trainingMetrics.IoU50  // Use IoU50 as the primary metric

        let validationMetrics = detector.validationMetrics.meanAveragePrecision
        var validationMAPValue: Double? = nil
        if !validationMetrics.IoU50.isNaN {
            validationMAPValue = validationMetrics.IoU50
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Object detector trained with CreateML CLI",
            version: "1.0"
        )

        try detector.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        // Get class labels from annotations
        let classLabels = try getClassLabels(from: annotationsURL)

        return TrainingResult(
            modelURL: outputURL,
            trainingMAP: trainingMAPValue,
            validationMAP: validationMAPValue,
            trainingDuration: trainingDuration,
            classLabels: classLabels
        )
    }

    private func getClassLabels(from annotationsURL: URL) throws -> [String] {
        let data = try Data(contentsOf: annotationsURL)
        guard let annotations = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }

        var labels = Set<String>()
        for annotation in annotations {
            if let objects = annotation["annotations"] as? [[String: Any]] {
                for obj in objects {
                    if let label = obj["label"] as? String {
                        labels.insert(label)
                    }
                }
            }
        }
        return Array(labels).sorted()
    }

    public enum TrainingError: LocalizedError {
        case missingAnnotations(String)

        public var errorDescription: String? {
            switch self {
            case .missingAnnotations(let message):
                return message
            }
        }
    }
}
