// test_public_backend_inference.mjs
import fs from 'fs';

const PUBLIC_BACKEND = 'https://importantly-switches-reported-hours.trycloudflare.com';

async function run() {
  console.log('1. Health check on public backend:', PUBLIC_BACKEND);
  const healthRes = await fetch(`${PUBLIC_BACKEND}/health`);
  const health = await healthRes.json();
  console.log('Health response:', health);
  if (!health.model?.loaded) throw new Error('Model not loaded on public backend');

  console.log('2. Uploading real test sonar tile to public backend...');
  const filePath = 'frontend/public/demo_samples/pipe_1693569383.780_x3500.jpg';
  const fileBytes = fs.readFileSync(filePath);
  const blob = new Blob([fileBytes], { type: 'image/jpeg' });
  const form = new FormData();
  form.append('file', blob, 'pipe_1693569383.780_x3500.jpg');

  const uploadRes = await fetch(`${PUBLIC_BACKEND}/api/v1/uploads/image`, {
    method: 'POST',
    body: form,
  });
  if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status} ${await uploadRes.text()}`);
  const uploadData = await uploadRes.json();
  console.log('Upload success, image_id:', uploadData.image_id);

  console.log('3. Running real YOLOv8n inference on public backend...');
  const inferRes = await fetch(`${PUBLIC_BACKEND}/api/v1/detections/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: uploadData.image_id }),
  });
  if (!inferRes.ok) throw new Error(`Inference failed: ${inferRes.status} ${await inferRes.text()}`);
  const inferData = await inferRes.json();
  console.log('Inference success! Model Version:', inferData.model_version);
  console.log('Detections count:', inferData.detections?.length);
  for (const det of inferData.detections || []) {
    console.log(` - Object: ${det.class_name}, Raw: ${det.confidence_raw}, Final: ${det.confidence_final}, Status: ${det.status}`);
  }

  console.log('4. Testing survey demo upload & batch processing on public backend...');
  const zipBytes = fs.readFileSync('frontend/public/demo_samples/geo_survey_demo.zip');
  const zipBlob = new Blob([zipBytes], { type: 'application/zip' });
  const zipForm = new FormData();
  zipForm.append('file', zipBlob, 'geo_survey_demo.zip');

  const surveyUpRes = await fetch(`${PUBLIC_BACKEND}/api/v1/uploads/survey`, {
    method: 'POST',
    body: zipForm,
  });
  if (!surveyUpRes.ok) throw new Error(`Survey upload failed: ${surveyUpRes.status} ${await surveyUpRes.text()}`);
  const surveyData = await surveyUpRes.json();
  console.log('Survey upload success, survey_id:', surveyData.survey_id);

  const runSurveyRes = await fetch(`${PUBLIC_BACKEND}/api/v1/surveys/${surveyData.survey_id}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  if (!runSurveyRes.ok) throw new Error(`Survey run failed: ${runSurveyRes.status} ${await runSurveyRes.text()}`);
  const jobData = await runSurveyRes.json();
  console.log('Survey batch job launched:', jobData.job_id);

  // Poll job
  let finished = false;
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 500));
    const jobStatusRes = await fetch(`${PUBLIC_BACKEND}/api/v1/jobs/${jobData.job_id}`);
    const statusObj = await jobStatusRes.json();
    if (statusObj.status === 'succeeded') {
      console.log('Survey batch job succeeded!');
      finished = true;
      break;
    }
  }
  if (!finished) throw new Error('Survey job timed out');

  // Fetch detections
  const detRes = await fetch(`${PUBLIC_BACKEND}/api/v1/detections?survey_id=${surveyData.survey_id}&size=100`);
  const detList = await detRes.json();
  console.log('Survey detections retrieved:', detList.items?.length);
  for (const d of detList.items || []) {
    console.log(` - Object: ${d.class_name}, Lat: ${d.latitude}, Lon: ${d.longitude}, Uncertainty: ±${d.uncertainty_m}m, Status: ${d.status}`);
  }

  console.log('\n===========================================');
  console.log('ALL PUBLIC BACKEND INFERENCE & SURVEY CHECKS PASSED!');
  console.log('===========================================');
}

run().catch((err) => {
  console.error('FAILED:', err);
  process.exit(1);
});
