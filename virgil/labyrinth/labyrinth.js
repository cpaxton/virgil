let roomsData = {};

function generateRoomHTML(roomData) {
    const room = document.createElement('div');
    room.className = 'room';

    const image = document.createElement('img');
    image.src = roomData.image_filename;
    image.alt = roomData.image;
    room.appendChild(image);

    const text = document.createElement('p');
    text.textContent = roomData.text;
    room.appendChild(text);

    const actionContainer = document.createElement('div');
    actionContainer.className = 'actions';
    
    for (const [targetRoom, actionText] of Object.entries(roomData.actions)) {
        const button = document.createElement('button');
        button.textContent = actionText;
        button.onclick = () => navigateToRoom(targetRoom);
        actionContainer.appendChild(button);
    }
    
    room.appendChild(actionContainer);

    return room;
}

function navigateToRoom(roomId) {
    displayRoom(roomId);
}

function displayRoom(roomId) {
    const container = document.getElementById('room-container');
    container.innerHTML = '';
    
    if (roomsData[roomId]) {
        const roomHTML = generateRoomHTML(roomsData[roomId]);
        container.appendChild(roomHTML);
    } else {
        container.textContent = 'Room not found';
    }
}

function loadRoomData() {
    fetch('descriptions.yaml')
        .then(response => response.text())
        .then(yamlText => {
            roomsData = jsyaml.load(yamlText);
            // Start with the first room (assuming '2_1' is the starting room)
            displayRoom('1_1');
        })
        .catch(error => console.error('Error loading room data:', error));
}

// Load room data when the page loads
window.onload = loadRoomData;

