import XCTest
@testable import CreateMLToolkit

final class CreateMLToolkitTests: XCTestCase {
    func testImageClassifierTrainerInit() throws {
        let trainer = ImageClassifierTrainer()
        XCTAssertNotNil(trainer)
    }

    func testTextClassifierTrainerInit() throws {
        let trainer = TextClassifierTrainer()
        XCTAssertNotNil(trainer)
    }

    func testSoundClassifierTrainerInit() throws {
        let trainer = SoundClassifierTrainer()
        XCTAssertNotNil(trainer)
    }

    func testTabularTrainerInit() throws {
        let trainer = TabularTrainer()
        XCTAssertNotNil(trainer)
    }

    func testTrainingParametersDefaults() throws {
        let imageParams = ImageClassifierTrainer.TrainingParameters()
        XCTAssertEqual(imageParams.maxIterations, 25)

        let textParams = TextClassifierTrainer.TrainingParameters()
        XCTAssertNil(textParams.validationData)

        let soundParams = SoundClassifierTrainer.TrainingParameters()
        XCTAssertEqual(soundParams.overlapFactor, 0.5)
    }
}
