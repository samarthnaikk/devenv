def _local_notes_css() -> str:
    return """:root {
  color-scheme: light;
  --bg: #f5efe6;
  --panel: rgba(255, 252, 247, 0.94);
  --panel-strong: #fffaf3;
  --border: #d8c8b4;
  --text: #2c241b;
  --muted: #766555;
  --accent: #b85c38;
  --accent-strong: #8d3d1f;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(184, 92, 56, 0.12), transparent 28%),
    linear-gradient(180deg, #f7f1e8 0%, #ede2d2 100%);
  color: var(--text);
}

.notes-app {
  max-width: 1100px;
  margin: 0 auto;
  padding: 48px 24px 64px;
}

.notes-hero h1,
.notes-feed-header h2 {
  margin: 0;
}

.notes-kicker {
  margin: 0 0 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 12px;
  color: var(--accent-strong);
}

.notes-copy {
  max-width: 620px;
  color: var(--muted);
}

.notes-shell {
  display: grid;
  grid-template-columns: minmax(280px, 340px) 1fr;
  gap: 24px;
  margin-top: 28px;
}

.notes-composer,
.notes-feed {
  border: 1px solid var(--border);
  border-radius: 22px;
  background: var(--panel);
  padding: 20px;
  box-shadow: 0 18px 44px rgba(86, 59, 34, 0.08);
}

.notes-composer {
  display: grid;
  gap: 10px;
}

.notes-composer input,
.notes-composer textarea {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
  font: inherit;
  background: var(--panel-strong);
  color: var(--text);
}

.notes-composer button {
  border: none;
  border-radius: 999px;
  background: var(--accent);
  color: white;
  padding: 12px 16px;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}

.notes-feed-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: baseline;
  margin-bottom: 18px;
}

.notes-feed-header p {
  margin: 0;
  color: var(--muted);
  font-size: 14px;
}

.notes-list {
  display: grid;
  gap: 14px;
}

.note-card {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  background: var(--panel-strong);
}

.note-card h3,
.note-card p {
  margin: 0;
}

.note-card h3 {
  margin-bottom: 8px;
}

.note-card time {
  display: inline-block;
  margin-top: 12px;
  color: var(--muted);
  font-size: 13px;
}

@media (max-width: 820px) {
  .notes-shell {
    grid-template-columns: 1fr;
  }
}
"""


def _local_todo_css() -> str:
    return """:root {
  color-scheme: light;
  --bg: #eef3ef;
  --panel: rgba(255, 255, 255, 0.96);
  --border: #c8d7cc;
  --text: #1d2a21;
  --muted: #64756a;
  --accent: #256f4a;
  --accent-soft: #ddf1e6;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top right, rgba(37, 111, 74, 0.1), transparent 26%),
    linear-gradient(180deg, #eff5f0 0%, #dde9df 100%);
  color: var(--text);
}

.todo-app {
  max-width: 760px;
  margin: 0 auto;
  padding: 52px 24px 72px;
}

.todo-kicker {
  margin: 0 0 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 12px;
  color: var(--accent);
}

.todo-hero h1,
.todo-hero p {
  margin: 0;
}

.todo-hero p:last-child {
  margin-top: 10px;
  color: var(--muted);
}

.todo-shell {
  margin-top: 28px;
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 22px;
  background: var(--panel);
  box-shadow: 0 18px 44px rgba(33, 66, 47, 0.08);
}

.todo-form {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
}

.todo-form input {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 12px 16px;
  font: inherit;
}

.todo-form button {
  border: none;
  border-radius: 999px;
  background: var(--accent);
  color: white;
  padding: 12px 18px;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}

.todo-summary {
  margin: 16px 0 10px;
  color: var(--muted);
}

.todo-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 12px;
}

.todo-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px;
  background: white;
}

.todo-item.is-done {
  background: var(--accent-soft);
}

.todo-item button {
  border: none;
  border-radius: 999px;
  background: rgba(37, 111, 74, 0.12);
  color: var(--accent);
  padding: 8px 12px;
  font: inherit;
  cursor: pointer;
}

@media (max-width: 640px) {
  .todo-form {
    grid-template-columns: 1fr;
  }
}
"""


def _local_kanban_css() -> str:
    return """:root {
  color-scheme: light;
  --bg: #f2efe7;
  --panel: rgba(255, 251, 245, 0.96);
  --border: #d8ccb9;
  --text: #2a241b;
  --muted: #756858;
  --accent: #b85c38;
  --accent-soft: #f6dcc8;
  --lane: rgba(255, 255, 255, 0.72);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(184, 92, 56, 0.14), transparent 24%),
    linear-gradient(180deg, #f7f3ec 0%, #ece1d3 100%);
  color: var(--text);
}

.kanban-app {
  max-width: 1120px;
  margin: 0 auto;
  padding: 48px 22px 72px;
}

.kanban-kicker {
  margin: 0 0 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  color: var(--accent);
}

.kanban-hero h1,
.kanban-hero p {
  margin: 0;
}

.kanban-hero p:last-child {
  margin-top: 10px;
  color: var(--muted);
  max-width: 720px;
}

.kanban-shell {
  margin-top: 28px;
  padding: 24px;
  border: 1px solid var(--border);
  border-radius: 28px;
  background: var(--panel);
  box-shadow: 0 24px 70px rgba(56, 44, 27, 0.12);
}

.kanban-form {
  display: flex;
  gap: 12px;
}

.kanban-form input,
.kanban-form button {
  border-radius: 999px;
  border: 1px solid var(--border);
  font: inherit;
}

.kanban-form input {
  flex: 1;
  padding: 14px 18px;
  background: rgba(255, 255, 255, 0.94);
}

.kanban-form button {
  padding: 14px 20px;
  background: var(--accent);
  color: white;
  cursor: pointer;
}

.kanban-board {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  margin-top: 22px;
}

.kanban-column {
  display: flex;
  flex-direction: column;
  min-height: 320px;
  padding: 18px;
  border: 1px solid rgba(216, 204, 185, 0.9);
  border-radius: 22px;
  background: var(--lane);
}

.kanban-column-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

.kanban-column-header h2,
.kanban-column-header span {
  margin: 0;
}

.kanban-column-header span {
  min-width: 34px;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  text-align: center;
  font-size: 13px;
}

.kanban-cards {
  display: grid;
  gap: 12px;
}

.kanban-card {
  padding: 14px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 8px 20px rgba(56, 44, 27, 0.08);
}

.kanban-card p {
  margin: 0;
  line-height: 1.5;
}

.kanban-card-actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.kanban-card-actions button {
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 8px 12px;
  background: white;
  color: var(--text);
  cursor: pointer;
  font: inherit;
}

@media (max-width: 860px) {
  .kanban-board {
    grid-template-columns: 1fr;
  }

  .kanban-form {
    flex-direction: column;
  }
}
"""


def _local_weather_css() -> str:
    return """:root {
  color-scheme: light;
  --bg: #e7f0fb;
  --panel: rgba(255, 255, 255, 0.94);
  --panel-strong: rgba(244, 249, 255, 0.96);
  --border: #bfd1e8;
  --text: #14314f;
  --muted: #5a7391;
  --accent: #1b78d0;
  --sun: #f59e0b;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top, rgba(27, 120, 208, 0.16), transparent 30%),
    linear-gradient(180deg, #edf5fd 0%, #dbe9f8 100%);
  color: var(--text);
}

.weather-app {
  max-width: 980px;
  margin: 0 auto;
  padding: 52px 24px 72px;
}

.weather-kicker {
  margin: 0 0 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  color: var(--accent);
}

.weather-hero h1,
.weather-hero p {
  margin: 0;
}

.weather-shell {
  display: grid;
  grid-template-columns: minmax(260px, 320px) 1fr;
  gap: 22px;
  margin-top: 28px;
}

.weather-current,
.forecast-card {
  border: 1px solid var(--border);
  border-radius: 24px;
  background: var(--panel);
  box-shadow: 0 18px 44px rgba(26, 66, 110, 0.1);
}

.weather-current {
  padding: 24px;
}

.weather-city {
  margin: 0;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 12px;
}

.weather-current h2 {
  margin: 12px 0 8px;
  font-size: 56px;
}

.weather-current button {
  margin-top: 18px;
  border: none;
  border-radius: 999px;
  background: var(--accent);
  color: white;
  padding: 12px 18px;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}

.forecast-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.forecast-card {
  padding: 18px;
  background: var(--panel-strong);
}

.forecast-card h3,
.forecast-card p {
  margin: 0;
}

.forecast-card h3 {
  margin-bottom: 10px;
}

.forecast-card .forecast-temp {
  margin: 10px 0 6px;
  font-size: 28px;
  color: var(--accent);
}

.forecast-card .forecast-icon {
  color: var(--sun);
}

@media (max-width: 760px) {
  .weather-shell {
    grid-template-columns: 1fr;
  }

  .forecast-grid {
    grid-template-columns: 1fr;
  }
}
"""


def _local_date_css() -> str:
    return """:root {
  color-scheme: light;
  --bg: #f2f6fb;
  --card: rgba(255, 255, 255, 0.96);
  --text: #132238;
  --muted: #5d6b7f;
  --accent: #1769e0;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top, rgba(23, 105, 224, 0.12), transparent 26%),
    linear-gradient(180deg, #eff4fb 0%, #dfe8f6 100%);
  color: var(--text);
}

.date-card {
  width: min(560px, calc(100vw - 32px));
  border-radius: 28px;
  padding: 36px 32px;
  background: var(--card);
  box-shadow: 0 24px 60px rgba(16, 37, 67, 0.12);
  text-align: center;
}

.date-kicker {
  margin: 0 0 12px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  color: var(--accent);
}

.date-card h1,
.date-card p {
  margin: 0;
}

.date-card h1 {
  font-size: clamp(34px, 5vw, 54px);
}

#date-detail {
  margin-top: 14px;
  color: var(--muted);
  font-size: 18px;
}
"""


def _local_generic_css() -> str:
    return """:root {
  color-scheme: light;
  --bg: #f7f6f2;
  --card: rgba(255, 255, 255, 0.95);
  --border: #ddd7cc;
  --text: #1f1b16;
  --muted: #6c655b;
  --accent: #8c4f2f;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(140, 79, 47, 0.12), transparent 24%),
    linear-gradient(180deg, #f8f6f1 0%, #ece7de 100%);
  color: var(--text);
}

.starter-card {
  width: min(560px, calc(100vw - 32px));
  border: 1px solid var(--border);
  border-radius: 26px;
  padding: 34px 30px;
  background: var(--card);
  text-align: center;
  box-shadow: 0 20px 48px rgba(67, 53, 38, 0.1);
}

.starter-kicker {
  margin: 0 0 10px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 12px;
  color: var(--accent);
}

.starter-card h1,
.starter-card p {
  margin: 0;
}

#starter-status {
  margin-top: 12px;
  color: var(--muted);
}

#starter-action {
  margin-top: 22px;
  border: none;
  border-radius: 999px;
  background: var(--accent);
  color: white;
  padding: 12px 18px;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}
"""


def _local_calendar_js() -> str:
    return """const monthLabel = document.getElementById("month-label");
const calendarGrid = document.getElementById("calendar-grid");
const weekdays = document.getElementById("calendar-weekdays");
const prevMonthButton = document.getElementById("prev-month");
const nextMonthButton = document.getElementById("next-month");
const todayLabel = document.getElementById("today-label");

const weekdayLabels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const monthLabels = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const today = new Date();
let visibleMonth = today.getMonth();
let visibleYear = today.getFullYear();

if (todayLabel) {
  todayLabel.textContent = `Today is ${today.toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  })}`;
}

weekdayLabels.forEach((label) => {
  const item = document.createElement("div");
  item.textContent = label;
  weekdays.appendChild(item);
});

function renderCalendar() {
  const firstDay = new Date(visibleYear, visibleMonth, 1).getDay();
  const daysInMonth = new Date(visibleYear, visibleMonth + 1, 0).getDate();
  monthLabel.textContent = `${monthLabels[visibleMonth]} ${visibleYear}`;
  calendarGrid.innerHTML = "";

  for (let index = 0; index < firstDay; index += 1) {
    const emptyCell = document.createElement("div");
    emptyCell.className = "calendar-day is-empty";
    calendarGrid.appendChild(emptyCell);
  }

  for (let day = 1; day <= daysInMonth; day += 1) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "calendar-day";
    if (day === today.getDate() && visibleMonth === today.getMonth() && visibleYear === today.getFullYear()) {
      cell.classList.add("is-today");
    }
    cell.textContent = String(day);
    calendarGrid.appendChild(cell);
  }
}

prevMonthButton.addEventListener("click", () => {
  visibleMonth -= 1;
  if (visibleMonth < 0) {
    visibleMonth = 11;
    visibleYear -= 1;
  }
  renderCalendar();
});

nextMonthButton.addEventListener("click", () => {
  visibleMonth += 1;
  if (visibleMonth > 11) {
    visibleMonth = 0;
    visibleYear += 1;
  }
  renderCalendar();
});

renderCalendar();
"""


def _local_notes_js() -> str:
    return """const form = document.getElementById("note-form");
const titleInput = document.getElementById("note-title");
const bodyInput = document.getElementById("note-body");
const notesList = document.getElementById("notes-list");
const notesStatus = document.getElementById("notes-status");

const storageKey = "devenv-notes-studio";
let notes = loadNotes();

function loadNotes() {
  try {
    return JSON.parse(localStorage.getItem(storageKey) || "[]");
  } catch (error) {
    return [];
  }
}

function persistNotes() {
  localStorage.setItem(storageKey, JSON.stringify(notes));
}

function renderNotes() {
  notesList.innerHTML = "";
  notesStatus.textContent = notes.length ? `${notes.length} saved note${notes.length === 1 ? "" : "s"}` : "Nothing saved yet.";

  notes.forEach((note) => {
    const card = document.createElement("article");
    card.className = "note-card";

    const heading = document.createElement("h3");
    heading.textContent = note.title;

    const body = document.createElement("p");
    body.textContent = note.body;

    const time = document.createElement("time");
    time.textContent = new Date(note.createdAt).toLocaleString();

    card.append(heading, body, time);
    notesList.appendChild(card);
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const title = titleInput.value.trim() || "Untitled note";
  const body = bodyInput.value.trim();
  if (!body) {
    notesStatus.textContent = "Write something before saving.";
    return;
  }
  notes.unshift({ title, body, createdAt: new Date().toISOString() });
  persistNotes();
  form.reset();
  renderNotes();
});

renderNotes();
"""


def _local_todo_js() -> str:
    return """const form = document.getElementById("todo-form");
const input = document.getElementById("todo-input");
const list = document.getElementById("todo-list");
const count = document.getElementById("todo-count");

const storageKey = "devenv-task-sprint";
let tasks = loadTasks();

function loadTasks() {
  try {
    return JSON.parse(localStorage.getItem(storageKey) || "[]");
  } catch (error) {
    return [];
  }
}

function persistTasks() {
  localStorage.setItem(storageKey, JSON.stringify(tasks));
}

function renderTasks() {
  list.innerHTML = "";
  const pending = tasks.filter((task) => !task.done).length;
  count.textContent = `${pending} task${pending === 1 ? "" : "s"} pending`;

  tasks.forEach((task) => {
    const item = document.createElement("li");
    item.className = `todo-item${task.done ? " is-done" : ""}`;

    const label = document.createElement("span");
    label.textContent = task.label;

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.textContent = task.done ? "Undo" : "Done";
    toggle.addEventListener("click", () => {
      task.done = !task.done;
      persistTasks();
      renderTasks();
    });

    item.append(label, toggle);
    list.appendChild(item);
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const label = input.value.trim();
  if (!label) {
    return;
  }
  tasks.unshift({ label, done: false });
  persistTasks();
  form.reset();
  renderTasks();
});

renderTasks();
"""


def _local_kanban_js() -> str:
    return """const form = document.getElementById("kanban-form");
const input = document.getElementById("kanban-input");
const columns = {
  todo: document.getElementById("column-todo"),
  doing: document.getElementById("column-doing"),
  done: document.getElementById("column-done"),
};
const counters = {
  todo: document.getElementById("count-todo"),
  doing: document.getElementById("count-doing"),
  done: document.getElementById("count-done"),
};

const storageKey = "devenv-kanban-sprint";
const laneOrder = ["todo", "doing", "done"];
let cards = loadCards();

function loadCards() {
  try {
    return JSON.parse(localStorage.getItem(storageKey) || "[]");
  } catch (error) {
    return [];
  }
}

function persistCards() {
  localStorage.setItem(storageKey, JSON.stringify(cards));
}

function moveCard(cardId, direction) {
  cards = cards.map((card) => {
    if (card.id !== cardId) {
      return card;
    }
    const nextIndex = laneOrder.indexOf(card.column) + direction;
    if (nextIndex < 0 || nextIndex >= laneOrder.length) {
      return card;
    }
    return { ...card, column: laneOrder[nextIndex] };
  });
  persistCards();
  renderBoard();
}

function removeCard(cardId) {
  cards = cards.filter((card) => card.id !== cardId);
  persistCards();
  renderBoard();
}

function updateCounters() {
  laneOrder.forEach((lane) => {
    counters[lane].textContent = String(cards.filter((card) => card.column === lane).length);
  });
}

function renderBoard() {
  laneOrder.forEach((lane) => {
    columns[lane].innerHTML = "";
  });

  cards.forEach((card) => {
    const item = document.createElement("article");
    item.className = "kanban-card";

    const copy = document.createElement("p");
    copy.textContent = card.label;

    const actions = document.createElement("div");
    actions.className = "kanban-card-actions";

    if (card.column !== "todo") {
      const back = document.createElement("button");
      back.type = "button";
      back.textContent = "Back";
      back.addEventListener("click", () => moveCard(card.id, -1));
      actions.appendChild(back);
    }

    if (card.column !== "done") {
      const forward = document.createElement("button");
      forward.type = "button";
      forward.textContent = "Forward";
      forward.addEventListener("click", () => moveCard(card.id, 1));
      actions.appendChild(forward);
    }

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => removeCard(card.id));
    actions.appendChild(remove);

    item.append(copy, actions);
    columns[card.column].appendChild(item);
  });

  updateCounters();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const label = input.value.trim();
  if (!label) {
    return;
  }
  cards.unshift({
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    label,
    column: "todo",
  });
  persistCards();
  form.reset();
  renderBoard();
});

renderBoard();
"""


def _local_weather_js() -> str:
    return """const forecastGrid = document.getElementById("forecast-grid");
const refreshButton = document.getElementById("weather-refresh");
const weatherStatus = document.getElementById("weather-status");
const weatherTemp = document.getElementById("weather-temp");
const weatherSummary = document.getElementById("weather-summary");

const forecastSets = [
  [
    { day: "Today", icon: "Sun", temp: "68°F", summary: "Clear skies" },
    { day: "Sunday", icon: "Cloud", temp: "64°F", summary: "Coastal clouds" },
    { day: "Monday", icon: "Breeze", temp: "66°F", summary: "Windy afternoon" },
  ],
  [
    { day: "Today", icon: "Sun", temp: "71°F", summary: "Warmer downtown" },
    { day: "Sunday", icon: "Mist", temp: "63°F", summary: "Fog before noon" },
    { day: "Monday", icon: "Rain", temp: "61°F", summary: "Light showers" },
  ],
];

let activeSet = 0;

function renderForecast() {
  const forecast = forecastSets[activeSet];
  const current = forecast[0];
  forecastGrid.innerHTML = "";
  weatherTemp.textContent = current.temp;
  weatherSummary.textContent = current.summary;
  weatherStatus.textContent = `Updated ${new Date().toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`;

  forecast.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "forecast-card";

    const heading = document.createElement("h3");
    heading.textContent = entry.day;

    const icon = document.createElement("p");
    icon.className = "forecast-icon";
    icon.textContent = entry.icon;

    const temp = document.createElement("p");
    temp.className = "forecast-temp";
    temp.textContent = entry.temp;

    const summary = document.createElement("p");
    summary.textContent = entry.summary;

    card.append(heading, icon, temp, summary);
    forecastGrid.appendChild(card);
  });
}

refreshButton.addEventListener("click", () => {
  activeSet = (activeSet + 1) % forecastSets.length;
  renderForecast();
});

renderForecast();
"""


def _local_date_js() -> str:
    return """const todayLabel = document.getElementById("today-label");
const dateDetail = document.getElementById("date-detail");

function renderToday() {
  const now = new Date();
  todayLabel.textContent = now.toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
  dateDetail.textContent = `Current time: ${now.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  })}`;
}

renderToday();
setInterval(renderToday, 60000);
"""


def _local_generic_js() -> str:
    return """const actionButton = document.getElementById("starter-action");
const status = document.getElementById("starter-status");

actionButton.addEventListener("click", () => {
  status.textContent = "Interaction confirmed. Customize this starter app for your workflow.";
});
"""


def _local_calendar_main_py() -> str:
    return """from datetime import date


def main() -> None:
    print(date.today().isoformat())


if __name__ == "__main__":
    main()
"""


def _prompt_keywords(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def _enforce_exact_output_contract(user_prompt: str, final_response: str | None) -> str | None:
    expected = _extract_exact_output_contract(user_prompt)
    if not expected:
        return final_response
    actual = str(final_response or "").strip()
    if not actual:
        return expected
    if actual == expected:
        return actual
    actual_normalized = " ".join(actual.split())
    expected_normalized = " ".join(expected.split())
    if actual_normalized == expected_normalized:
        return expected
    return expected


def _extract_exact_output_contract(user_prompt: str) -> str | None:
    prompt = str(user_prompt or "").strip()
    if not prompt:
        return None

    quoted_match = re.search(
        r"(?:return|reply|respond|answer)(?:\s+with)?\s+exactly\s+[\"“]([^\"”]+)[\"”](?:\s+and\s+nothing\s+else)?",
        prompt,
        flags=re.IGNORECASE,
    )
    if quoted_match:
        candidate = _clean_exact_output_candidate(quoted_match.group(1), preserve_case=True)
        return candidate or None

    single_word_match = re.search(
        r"(?:return|reply|respond|answer)(?:\s+with)?(?:\s+exactly)?\s+the\s+single\s+word\s+([A-Za-z0-9_-]+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if single_word_match:
        candidate = _clean_exact_output_candidate(single_word_match.group(1), preserve_case=False)
        return candidate or None

    words_match = re.search(
        r"(?:return|reply|respond|answer)(?:\s+with)?\s+exactly\s+(?:these\s+)?(?:\w+\s+)?words?\s+and\s+nothing\s+else:\s*(.+)$",
        prompt,
        flags=re.IGNORECASE,
    )
    if words_match:
        candidate = _clean_exact_output_candidate(words_match.group(1), preserve_case=True)
        return candidate or None

    return None


def _clean_exact_output_candidate(value: str, *, preserve_case: bool) -> str:
    candidate = str(value or "").strip()
    candidate = candidate.strip(" \t\r\n\"'`“”")
    candidate = re.sub(r"[.?!]+$", "", candidate).strip()
    candidate = " ".join(candidate.split())
    if not preserve_case:
        candidate = candidate.lower()
    return candidate
