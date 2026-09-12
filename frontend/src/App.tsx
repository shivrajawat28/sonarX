import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import WorkbenchPage from "./pages/WorkbenchPage";
import SurveyPage from "./pages/SurveyPage";
import HistoryPage from "./pages/HistoryPage";
import ModelsPage from "./pages/ModelsPage";

/**
 * Top-level routes. The frontend contains NO ML logic — pages only fetch,
 * display, and send user actions (upload / run / save / export) to the API.
 */
export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<WorkbenchPage />} />
        <Route path="/survey" element={<SurveyPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/models" element={<ModelsPage />} />
      </Routes>
    </Layout>
  );
}
