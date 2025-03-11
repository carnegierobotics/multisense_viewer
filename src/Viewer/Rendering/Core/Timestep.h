//
// Created by magnus on 3/11/25.
//

#ifndef TIMESTEP_H
#define TIMESTEP_H

namespace VkRender {
    class Timestep {
    public:
        Timestep(float time = 0.0f) : m_time(time) {};


        float getSeconds() const { return m_time; }
        float getMilliSeconds() const { return m_time * 1000.0f; }

        explicit operator float() const { return m_time; }

        // Overload multiplication operator: Timestep * float
        Timestep operator*(float multiplier) const {
            return Timestep(m_time * multiplier);
        }

        // Optionally, overload multiplication assignment operator
        Timestep& operator*=(float multiplier) {
            m_time *= multiplier;
            return *this;
        }

        // Multiply a scalar by a Timestep, returning a float.
        friend float operator*(float multiplier, const Timestep& ts) {
            return multiplier * ts.m_time;
        }

    private:
        float m_time;
    };
}

#endif //TIMESTEP_H
