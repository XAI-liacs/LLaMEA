document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const roleInput = document.getElementById('roleInput');
    const sendButton = document.getElementById('sendButton');

    socket.on('load_messages', (messages) => {
        messages.forEach(addMessage);
    });

    socket.on('new_message', (message) => {
        addMessage(message);
    });

    sendButton.addEventListener('click', () => {
        const messageText = messageInput.value;
        if (messageText.trim()) {
            const message = { role: roleInput.value, content: messageText };
            socket.emit('new_message', message);
            messageInput.value = '';
        }
    });

    function addMessage(message) {
        var messageContent = marked.parse(message.content)
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.innerHTML = `<span class="role">${message.role}:</span> ${messageContent} <span class="time">[${new Date(message.time).toLocaleTimeString()}]</span>`;
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});
