function phaseDist = phase_distance(dist)
    phaseDist = zeros(size(dist));
    phaseDist(dist<pi) = dist(dist<pi);
    phaseDist(dist>=pi) = 2*pi-dist(dist>=pi);
end