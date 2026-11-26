import LearningTab from "./Learning";

// Training that belongs to no mission: dataset versions of any missions (still images and video together), trained,
// compared and then put to use in whichever missions the operator chooses. The learning screens are shared with a
// mission's Learning tab; the server keeps this page's rounds and evaluations apart from every mission.
export default function TrainingPage() {
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Training</h1>
          <p>Train and compare vehicle detectors on dataset versions from any missions, then choose where each model is used.</p>
        </div>
      </div>
      <LearningTab missionId="training" job={null} reload={() => undefined} />
    </div>
  );
}
