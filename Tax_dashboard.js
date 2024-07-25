import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Button } from "@/components/ui/button"

const taxData = [
  { country: 'France', corporateTax: 25, corporateReductionMin: 3, corporateReductionMax: 8, individualTax: 45, individualReductionMin: 5, individualReductionMax: 15 },
  { country: 'Allemagne', corporateTax: 15.825, corporateReductionMin: 1, corporateReductionMax: 5, individualTax: 45, individualReductionMin: 4, individualReductionMax: 12 },
  { country: 'Royaume-Uni', corporateTax: 25, corporateReductionMin: 2, corporateReductionMax: 7, individualTax: 45, individualReductionMin: 3, individualReductionMax: 10 },
  { country: 'Italie', corporateTax: 24, corporateReductionMin: 2, corporateReductionMax: 6, individualTax: 43, individualReductionMin: 3, individualReductionMax: 9 },
  { country: 'Espagne', corporateTax: 25, corporateReductionMin: 2, corporateReductionMax: 7, individualTax: 45, individualReductionMin: 4, individualReductionMax: 11 },
  { country: 'Pays-Bas', corporateTax: 25.8, corporateReductionMin: 3, corporateReductionMax: 8, individualTax: 49.5, individualReductionMin: 5, individualReductionMax: 14 },
  { country: 'Suède', corporateTax: 20.6, corporateReductionMin: 1, corporateReductionMax: 4, individualTax: 52, individualReductionMin: 3, individualReductionMax: 8 },
  { country: 'États-Unis', corporateTax: 21, corporateReductionMin: 4, corporateReductionMax: 12, individualTax: 37, individualReductionMin: 6, individualReductionMax: 18 },
];

const TaxComparisonDashboard = () => {
  const [selectedCountries, setSelectedCountries] = useState(['France', 'États-Unis']);

  const toggleCountry = (country) => {
    setSelectedCountries(prev => 
      prev.includes(country)
        ? prev.filter(c => c !== country)
        : [...prev, country]
    );
  };

  const filteredData = taxData
    .filter(item => selectedCountries.includes(item.country))
    .map(item => ({
      country: item.country,
      "Taux entreprises (effectif min)": item.corporateTax - item.corporateReductionMax,
      "Réduction max. entreprises": item.corporateReductionMax - item.corporateReductionMin,
      "Réduction min. entreprises": item.corporateReductionMin,
      "Taux particuliers (effectif min)": item.individualTax - item.individualReductionMax,
      "Réduction max. particuliers": item.individualReductionMax - item.individualReductionMin,
      "Réduction min. particuliers": item.individualReductionMin,
      corporateTax: item.corporateTax,
      individualTax: item.individualTax,
    }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border rounded shadow">
          <p className="font-bold">{data.country}</p>
          <p>Taux d'imposition des entreprises (brut): {data.corporateTax}%</p>
          <p>Réduction minimale: {data["Réduction min. entreprises"]}%</p>
          <p>Réduction maximale: {data["Réduction min. entreprises"] + data["Réduction max. entreprises"]}%</p>
          <p>Taux effectif potentiel: {data["Taux entreprises (effectif min)"].toFixed(1)}% - {(data.corporateTax - data["Réduction min. entreprises"]).toFixed(1)}%</p>
          <p>Taux d'imposition max. des particuliers (brut): {data.individualTax}%</p>
          <p>Réduction minimale: {data["Réduction min. particuliers"]}%</p>
          <p>Réduction maximale: {data["Réduction min. particuliers"] + data["Réduction max. particuliers"]}%</p>
          <p>Taux effectif potentiel: {data["Taux particuliers (effectif min)"].toFixed(1)}% - {(data.individualTax - data["Réduction min. particuliers"]).toFixed(1)}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Comparaison des taux d'imposition et réductions (2023/2024)</h1>
      <div className="mb-4 flex flex-wrap gap-2">
        {taxData.map(item => (
          <Button
            key={item.country}
            onClick={() => toggleCountry(item.country)}
            variant={selectedCountries.includes(item.country) ? "default" : "outline"}
          >
            {item.country}
          </Button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={500}>
        <BarChart data={filteredData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="country" />
          <YAxis />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar dataKey="Taux entreprises (effectif min)" stackId="a" fill="#8884d8" />
          <Bar dataKey="Réduction max. entreprises" stackId="a" fill="#9c95e0" />
          <Bar dataKey="Réduction min. entreprises" stackId="a" fill="#b0a7f5" />
          <Bar dataKey="Taux particuliers (effectif min)" stackId="b" fill="#82ca9d" />
          <Bar dataKey="Réduction max. particuliers" stackId="b" fill="#95dca8" />
          <Bar dataKey="Réduction min. particuliers" stackId="b" fill="#a8eeb6" />
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-4 text-sm text-gray-600">
        <p>Note: Chaque barre représente le taux d'imposition total. La partie inférieure (couleur plus foncée) montre le taux effectif minimum après réductions maximales.</p>
        <p>La partie supérieure de chaque barre est divisée en deux sections : la réduction minimale (couleur la plus claire) et la réduction supplémentaire potentielle (couleur intermédiaire).</p>
        <p>Les réductions potentielles incluent les crédits d'impôt, les allègements de charges, etc. Les taux effectifs peuvent varier selon les situations individuelles.</p>
      </div>
    </div>
  );
};

export default TaxComparisonDashboard;
