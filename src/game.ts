export type VoiceLabel = 'LEFT' | 'RIGHT';

type Obstacle = { lane: number; y: number; w: number; h: number };

export class InfiniteRunnerGame {
    private ctx: CanvasRenderingContext2D;
    private width: number;
    private height: number;

    // lanes: 0 (left), 1 (center), 2 (right)
    private lanes = 3;
    private laneXs: number[] = [];
    private laneW = 100;

    private playerLane = 1;
    private playerY: number;
    private playerW = 70;
    private playerH = 70;

    private obstacles: Obstacle[] = [];
    private speed = 3; // px per frame
    private spawnEvery = 70; // frames
    private frame = 0;
    private running = false;
    private score = 0;

    private scoreEl: HTMLElement | null;
    private speedEl: HTMLElement | null;

    constructor(private canvas: HTMLCanvasElement) {
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('2D context not available');
        this.ctx = ctx;
        this.width = canvas.width;
        this.height = canvas.height;
        this.playerY = this.height - 110;

        const padding = (this.width - this.lanes * this.laneW) / (this.lanes + 1);
        for (let i = 0; i < this.lanes; i++) {
            this.laneXs.push(padding + i * (this.laneW + padding) + this.laneW / 2);
        }

        this.scoreEl = document.querySelector('#score');
        this.speedEl = document.querySelector('#speed');

        window.addEventListener('voice-command', ((e: CustomEvent) => {
            const label: VoiceLabel = e.detail.label;
            if (!this.running) return;
            if (label === 'LEFT') this.moveLeft();
            else if (label === 'RIGHT') this.moveRight();
        }) as EventListener);

        window.addEventListener('keydown', (e) => {
            if (!this.running) return;
            if (e.key === 'ArrowLeft') this.moveLeft();
            if (e.key === 'ArrowRight') this.moveRight();
        });
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.loop();
    }

    pause() { this.running = false; }

    reset() {
        this.running = false;
        this.obstacles = [];
        this.playerLane = 1;
        this.frame = 0;
        this.score = 0;
        this.speed = 3;
        this.spawnEvery = 70;
        this.draw();
        this.updateHUD();
    }

    private loop = () => {
        if (!this.running) return;
        this.update();
        this.draw();
        requestAnimationFrame(this.loop);
    };

    private update() {
        this.frame++;

        // Spawn obstacles
        if (this.frame % this.spawnEvery === 0) {
            const count = Math.random() < 0.75 ? 1 : 2; // sometimes two at once
            const lanes = this.sampleLanes(count);
            lanes.forEach((lane) => {
                this.obstacles.push({ lane, y: -80, w: 70, h: 70 });
            });
            // increase difficulty slowly
            if (this.spawnEvery > 35) this.spawnEvery--;
            if (this.frame % 600 === 0 && this.speed < 10) this.speed += 0.5;
        }

        // Move obstacles
        for (const o of this.obstacles) o.y += this.speed;

        // Remove off-screen
        this.obstacles = this.obstacles.filter((o) => o.y < this.height + 100);

        // Collision
        const px = this.laneXs[this.playerLane] - this.playerW / 2;
        const py = this.playerY;
        for (const o of this.obstacles) {
            const ox = this.laneXs[o.lane] - o.w / 2;
            const oy = o.y;
            if (px < ox + o.w && px + this.playerW > ox && py < oy + o.h && py + this.playerH > oy) {
                this.gameOver();
                return;
            }
        }

        // Score
        this.score += Math.floor(this.speed);
        this.updateHUD();
    }

    private draw() {
        const g = this.ctx;
        g.clearRect(0, 0, this.width, this.height);

        // lanes
        g.strokeStyle = '#333';
        g.lineWidth = 2;
        const laneColor = '#333';
        g.strokeStyle = laneColor;
        for (let i = 0; i < this.lanes; i++) {
            const x = this.laneXs[i];
            g.beginPath();
            g.moveTo(x, 0);
            g.lineTo(x, this.height);
            g.stroke();
        }

        // player
        const px = this.laneXs[this.playerLane] - this.playerW / 2;
        const py = this.playerY;
        g.fillStyle = '#4caf50';
        g.fillRect(px, py, this.playerW, this.playerH);

        // obstacles
        g.fillStyle = '#e53935';
        for (const o of this.obstacles) {
            const ox = this.laneXs[o.lane] - o.w / 2;
            g.fillRect(ox, o.y, o.w, o.h);
        }

        // ground gradient
        const grd = g.createLinearGradient(0, this.height - 120, 0, this.height);
        grd.addColorStop(0, 'rgba(255,255,255,0)');
        grd.addColorStop(1, 'rgba(255,255,255,0.05)');
        g.fillStyle = grd;
        g.fillRect(0, this.height - 120, this.width, 120);
    }

    private gameOver() {
        this.running = false;
        this.draw();
        const g = this.ctx;
        g.fillStyle = 'rgba(0,0,0,0.65)';
        g.fillRect(0, 0, this.width, this.height);
        g.fillStyle = '#fff';
        g.font = 'bold 28px system-ui';
        g.textAlign = 'center';
        g.fillText('Game Over', this.width / 2, this.height / 2 - 10);
        g.font = '16px system-ui';
        g.fillText(`Score: ${this.score}`, this.width / 2, this.height / 2 + 20);
    }

    private updateHUD() {
        if (this.scoreEl) this.scoreEl.textContent = String(this.score);
        if (this.speedEl) this.speedEl.textContent = this.speed.toFixed(1);
    }

    private sampleLanes(count: number): number[] {
        const all = [0, 1, 2];
        for (let i = all.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [all[i], all[j]] = [all[j], all[i]];
        }
        return all.slice(0, count);
    }

    moveLeft() { this.playerLane = Math.max(0, this.playerLane - 1); }
    moveRight() { this.playerLane = Math.min(this.lanes - 1, this.playerLane + 1); }
}
