import React from "https://esm.sh/react@18";
import { fetchFiles, fetchHealth, runTurn } from "./api.js";
import { FileSidebar } from "./components/FileSidebar.js";
import { HeaderBar } from "./components/HeaderBar.js";
import { StepsPanel } from "./components/StepsPanel.js";
import { TerminalPanel } from "./components/TerminalPanel.js";

export function App() {
  const [health, setHealth] = React.useState(null);
  const [entries, setEntries] = React.useState([]);
  const [selectedPath, setSelectedPath] = React.useState("");
  const [currentPath, setCurrentPath] = React.useState("");
  const [prompt, setPrompt] = React.useState("Tell me about this project.");
  const [transcript, setTranscript] = React.useState([]);
  const [steps, setSteps] = React.useState([]);
  const [usage, setUsage] = React.useState({});
  const [isRunning, setIsRunning] = React.useState(false);

  React.useEffect(() => {
    Promise.all([fetchHealth(), fetchFiles("")]).then(([healthPayload, filePayload]) => {
      setHealth(healthPayload);
      setEntries(filePayload.entries);
      setCurrentPath(filePayload.path);
    });
  }, []);

  if (!health) {
    return React.createElement("div", { className: "loading-shell" }, "Booting Devenv web interface...");
  }

  return React.createElement(
    "div",
    { className: "app-shell" },
    React.createElement(HeaderBar, {
      workspacePath: health.workspace_path,
      status: health.status,
    }),
    React.createElement(FileSidebar, {
      entries,
      activePath: selectedPath,
      currentPath,
      onSelectFile: setSelectedPath,
      onEnterDirectory: async (path) => {
        const filePayload = await fetchFiles(path);
        setEntries(filePayload.entries);
        setCurrentPath(filePayload.path);
      },
    }),
    React.createElement(TerminalPanel, {
      transcript,
      prompt,
      onPromptChange: setPrompt,
      isRunning,
      onSubmit: async () => {
        const nextPrompt = prompt.trim();
        if (!nextPrompt) {
          return;
        }
        setIsRunning(true);
        setTranscript((current) => [...current, { role: "user", content: nextPrompt }]);
        try {
          const result = await runTurn(nextPrompt);
          setTranscript((current) => [
            ...current,
            { role: "assistant", content: result.final_response || "No assistant response returned." },
          ]);
          setSteps(result.steps);
          setUsage(result.total_usage);
          setPrompt("");
        } catch (error) {
          setTranscript((current) => [
            ...current,
            { role: "assistant", content: `Request failed: ${error.message}` },
          ]);
        } finally {
          setIsRunning(false);
        }
      },
    }),
    React.createElement(StepsPanel, { steps, usage })
  );
}
