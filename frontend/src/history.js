// UI metadata must never enter the existing ChatMessage contract.
export function buildChatHistory(chat_history, messages) {
  const completedTurns = [];
  for (let index = 0; index < messages.length - 1; index += 1) {
    const user = messages[index];
    const assistant = messages[index + 1];
    if (user.role === "user" && assistant.role === "assistant" && assistant.status === "done" && assistant.content.trim()) {
      completedTurns.push(user, assistant);
      index += 1;
    }
  }
  return [...chat_history, ...completedTurns]
    .filter((message) => ["user", "assistant", "system"].includes(message.role) && typeof message.content === "string" && message.content.trim())
    .slice(-20)
    .map(({ role, content }) => ({ role, content: [...content].slice(0, 8000).join("") }));
}
